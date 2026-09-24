import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from FlagEmbedding import FlagReranker

# Safely import pre-configured components from the ingestion script
from chunking_vectordb import build_safe_context, retriever, token_length, client
from dynamic_splitter import expand_following_neighbors

from bm25_hybrid import get_hybrid_retriever


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
    # -------------------------------------------------------------------------
    # STAGE 1: BROAD CHILD RETRIEVAL
    # -------------------------------------------------------------------------
    # Query raw child chunks in Qdrant (Markdown AST Tier 5)
    # k=8 gives the bi-encoder sufficient breadth without memory bloat
    # child_docs = retriever.vectorstore.similarity_search(
    #     user_query,
    #     k=8
    # )

    child_docs = hybrid_child_retriever.invoke(user_query)

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

    # -------------------------------------------------------------------------
    # STAGE 5: FAITHFUL GENERATION
    # -------------------------------------------------------------------------
    formatted_prompt = STRICT_QA_PROMPT.format(
        context=safe_context,
        question=user_query
    )

    response = llm.invoke(formatted_prompt)
    answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    
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