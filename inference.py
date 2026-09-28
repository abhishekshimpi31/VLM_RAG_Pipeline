import logging
import os
import re
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import json

from FlagEmbedding import FlagReranker

# Safely import pre-configured components from the ingestion script
from chunking_vectordb import build_safe_context, retriever, token_length, client
from dynamic_splitter import expand_following_neighbors

from bm25_hybrid import get_hybrid_retriever



QUERY_EXPANSION_PROMPT = PromptTemplate.from_template(
    """You are a strict search optimization assistant for a professional retrieval-augmented generation (RAG) system.
Your task is to rewrite the user's search query to maximize retrieval accuracy from a vector database.

RULES:
1. Vocabulary Normalization: Translate colloquial visual or structural terms (e.g., "image", "picture", "graph", "chart", "page") into standard formal nomenclature (e.g., "Figure", "Table", "Section", "Appendix") based on typical professional formatting.
2. Exact Identifier Preservation: Keep all numbers, alphanumeric IDs, dates, and proper nouns (e.g., "3.20", "3.SM.1", "Q3", "John Doe") EXACTLY intact. 
3. Semantic Expansion: If the query is extremely sparse (e.g., "What does Figure 3.20 show?"), append broad contextual keywords like "data, findings, description, analysis" to increase search surface area.
4. Strict Context Adherence: You must only expand on the exact topic requested. You are strictly forbidden from attempting to answer the user's question or introducing external concepts. 
5. Output Format: Respond ONLY in valid JSON format with a single key "expanded_query".

USER QUERY:
{query}

JSON RESPONSE:"""
)

# ==============================================================================
#  HARDENED SCIENTIFIC PROMPT TEMPLATE (Faithfulness >= 0.95)
# ==============================================================================
PROMPT_TEMPLATE_TEXT = """You are an authoritative IPCC scientific assessment assistant.
Answer the user's question using ONLY the factual evidence provided in the context below.

CRITICAL INSTRUCTIONS:
1. STRICT ADHERENCE: Base your entire answer strictly on the provided context documents. Do not assume, extrapolate, or bring in outside knowledge.
2. EXACT FIGURES & CONFIDENCE: Quote quantitative values, anomalies, time intervals, and IPCC calibrated uncertainty terms (e.g., "high confidence", "very likely", "medium confidence") exactly as they appear.
3. CONFLICTS / GAPS: If the context does not explicitly provide the facts needed to answer the question, do not speculate. State exactly:
   "I do not have enough information in the provided context to answer this question."

--------------------
CONTEXT DOCUMENTS:
{context}
--------------------

QUESTION: {question}

SCIENTIFIC ANSWER:"""

STRICT_QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE_TEXT,
    input_variables=["context", "question"]
)

# ==============================================================================
# 1. LLM SETUP
# ==============================================================================
print("[INFO] Loading Ollama Llama-3.1...")
llm = Ollama(model="llama3.1", temperature=0.1, num_ctx=6000, verbose=False)

# ==============================================================================
# 2. RERANKER SETUP
# ==============================================================================
print("[INFO] Loading FlagReranker...")
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

# ==============================================================================
# 3. ROUTER & GENERATION PIPELINE
# ==============================================================================
def is_global_summary_query(query: str) -> bool:
    trigger_words = ["list all figures", "all images", "summary of all visual", "what images are in"]
    return any(trigger in query.lower() for trigger in trigger_words)


def expand_query(original_query: str) -> str:
    """
    Rewrites the user's query to normalize scientific vocabulary.
    Includes robust regex parsing to handle Llama-3 markdown quirks.
    """
    try:
        chain = QUERY_EXPANSION_PROMPT | llm
        response = chain.invoke({"query": original_query})

        print(response)
        
        # 1. Safely extract the raw string from the LLM
        raw_text = response.content if hasattr(response, "content") else str(response)
        
        # 2. Use regex to extract ONLY the JSON dictionary, ignoring any markdown backticks
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if not json_match:
            raise ValueError(f"No JSON object found in response: {raw_text}")
            
        clean_json_string = json_match.group(0)
        
        # 3. Parse the cleaned string
        response_data = json.loads(clean_json_string)
        expanded = response_data.get("expanded_query", original_query)
        
        logging.info(f"[QUERY EXPANSION] Original: '{original_query}' -> Expanded: '{expanded}'")
        return expanded
        
    except Exception as e:
        # Fallback to the original query if the LLM fails completely
        logging.warning(f"Query expansion failed: {e}. Using original query.")
        return original_query

# ==============================================================================
# 4. Initialize the fused hybrid retriever (k=8 per retriever, fused via RRF)
# ==============================================================================
hybrid_child_retriever = get_hybrid_retriever(k=8, vector_weight=0.5, bm25_weight=0.5)

def generate_answer(user_query: str):
    """
    Two-Stage Industrial RAG Inference:
    1. Broad child-vector search via Qdrant (k=8)
    2. Deep Cross-Attention reranking with BGE-Reranker-Large (Top-3)
    3. Docstore parent resolution (ParentDocumentRetriever)
    4. Sibling chunk expansion & token budgeting
    5. Grounded LLM generation
    
    Returns:
        tuple: (answer_text, safe_context, expanded_docs)
    """

    # 1. Expand the query to fix vocabulary mismatches
    optimized_query = expand_query(user_query)

    print(f"[INFO] Expanded Query: '{optimized_query}'")

    # -------------------------------------------------------------------------
    # STAGE 1: BROAD CHILD RETRIEVAL
    # -------------------------------------------------------------------------
    # Query raw child chunks in Qdrant (Markdown AST Tier 5)
    # k=8 gives the bi-encoder sufficient breadth without memory bloat
    # child_docs = retriever.vectorstore.similarity_search(
    #     optimized_query,
    #     k=10
    # )

    # print("child_docs", child_docs)

    child_docs = hybrid_child_retriever.invoke(optimized_query)

    if not child_docs:
        return "I do not have enough information in the provided context to answer this question.", "", []

    # -------------------------------------------------------------------------
    # STAGE 2: CROSS-ENCODER RERANKING (AT CHILD LEVEL)
    # -------------------------------------------------------------------------
    # Pairs fit comfortably inside BGE-Reranker's 512-token context limit
    pairs = [[user_query, doc.page_content] for doc in child_docs]
    scores = reranker.compute_score(pairs)
    
    if isinstance(scores, float):
        scores = [scores]

    # Rank children descending by cross-encoder logit scores
    scored_children = sorted(zip(child_docs, scores), key=lambda x: x[1], reverse=True)
    top_children = [doc for doc, score in scored_children[:3]]

    # -------------------------------------------------------------------------
    # STAGE 3: DEDUPLICATED PARENT RESOLUTION FROM DOCSTORE
    # -------------------------------------------------------------------------
    # Resolve the small child chunks back to their complete Parent AST sections
    id_key = getattr(retriever, "id_key", "doc_id")
    ordered_parent_ids = []
    seen_ids = set()

    for child in top_children:
        parent_id = child.metadata.get(id_key)
        if parent_id and parent_id not in seen_ids:
            seen_ids.add(parent_id)
            ordered_parent_ids.append(parent_id)

    # Fetch parent documents from the underlying docstore
    parent_docs = []
    if ordered_parent_ids:
        raw_parents = retriever.docstore.mget(ordered_parent_ids)
        parent_docs = [p for p in raw_parents if p is not None]

    # Fallback to children if parent lookup returns empty
    base_docs_for_expansion = parent_docs if parent_docs else top_children

    # -------------------------------------------------------------------------
    # STAGE 4: FORWARD NEIGHBOR EXPANSION & SAFE CONTEXT BUDGETING
    # -------------------------------------------------------------------------
    expanded_docs = [
        expand_following_neighbors(doc, retriever=retriever, min_tokens=1000)
        for doc in base_docs_for_expansion
    ]

    safe_context = build_safe_context(expanded_docs, max_tokens=5000)  # 10k token budget for LLM context

    print("safe_context", safe_context)

    # -------------------------------------------------------------------------
    # STAGE 5: FAITHFUL GENERATION
    # -------------------------------------------------------------------------
    formatted_prompt = STRICT_QA_PROMPT.format(
        context=safe_context,
        question=user_query
    )

    response = llm.invoke(formatted_prompt)
    answer_text = response.content if hasattr(response, "content") else str(response)
    # answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    print("answer_text", answer_text)

    
    return answer_text, safe_context, expanded_docs

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    user_query = "What does image 3.20 show?"
    try:
        generate_answer(user_query= user_query)
        
    finally:
        # Ensures clean database shutdown even if an error occurs
        print("\n[INFO] Closing database connections...")
        client.close()