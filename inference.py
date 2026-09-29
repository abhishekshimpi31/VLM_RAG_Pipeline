import logging
import os
import re
import json

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from FlagEmbedding import FlagReranker
from qdrant_client.http import models as rest

# Safely import pre-configured components from the ingestion script
from chunking_vectordb import build_safe_context, retriever, client, vector_store
from dynamic_splitter import expand_following_neighbors


# ==============================================================================
# PROMPTS
# ==============================================================================
QUERY_EXPANSION_PROMPT = PromptTemplate.from_template(
    """You are a strict search optimization assistant for a professional retrieval-augmented generation (RAG) system.
Your task is to evaluate and potentially rewrite the user's search query to maximize retrieval accuracy from a vector database.

RULES:
1. Conditional Modification: ONLY modify the query if it is overly sparse, lacks context, or uses colloquial terminology. If the query is already detailed, well-formulated, and uses proper terminology, return it EXACTLY as-is without any changes.
2. Vocabulary Normalization: Translate colloquial visual or structural terms (e.g., "image", "picture", "graph", "chart") into standard formal nomenclature (e.g., "Figure", "Table", "Section") based on typical professional formatting.
3. Exact Identifier Preservation: Keep all numbers, alphanumeric IDs, dates, and proper nouns (e.g., "3.20", "3.SM.1", "Q3", "John Doe") EXACTLY intact. 
4. Natural Semantic Expansion: If rewriting a sparse query (e.g., "What does Figure 3.20 show?"), do not just append disconnected keywords. Instead, rewrite it into a natural, context-rich sentence (e.g., "Describe the data, findings, and analytical context presented in Figure 3.20.").
5. Strict Context Adherence: You must only expand on the exact topic requested. You are strictly forbidden from attempting to answer the user's question, guessing the topic, or introducing external facts. 
6. Output Format: Respond ONLY in valid JSON format with a single key "expanded_query".

USER QUERY:
{query}

JSON RESPONSE:"""
)

PROMPT_TEMPLATE_TEXT = """You are an expert analytical assistant and strict fact-synthesizer. Your sole purpose is to extract and summarize information EXCLUSIVELY from the provided source documents.
You will be provided with a JSON array of source documents. Each document contains 'metadata', 'content', and a 'document_index'.

CRITICAL INSTRUCTIONS FOR MAXIMUM FAITHFULNESS:
1. ZERO OUTSIDE KNOWLEDGE: Your answer must be 100% grounded in the provided JSON context. Do not include any external knowledge, assumptions, logical leaps, or explanations that are not explicitly stated in the text. Even if you know a fact to be true, if it is not in the context, DO NOT mention it.
2. EXACT QUANTIFICATION: Quote all numerical values, percentages, dates, and calibrated uncertainty terms (e.g., "high confidence", "very likely") exactly as they appear in the source. Do not round numbers or approximate.
3. HANDLING MISSING INFORMATION: 
   - If the context completely lacks the facts needed to answer the question, you must state EXACTLY: "I do not have enough information in the provided context to answer this question."
   - If the context only partially answers the question, provide ONLY the information present in the text and do not guess the rest.
4. MANDATORY CITATIONS: Every single sentence or distinct factual claim you write MUST be immediately followed by its source citation using the 'document_index' in square brackets (e.g., [1]). 
   - STRICT RULE: If a sentence cannot be directly cited to the provided text, you are not allowed to write that sentence.
   - Example: "Global mean sea level increased by 0.20m between 1901 and 2018 [1]. This rate is faster than any preceding century in at least 3000 years [2]."

JSON CONTEXT:
{context}

USER QUESTION: 
{question}

Synthesize a direct, highly accurate answer based ONLY on the context above. Include citations for every claim:"""

STRICT_QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE_TEXT,
    input_variables=["context", "question"]
)

# ==============================================================================
# 1. LLM & RERANKER SETUP
# ==============================================================================
print("[INFO] Loading Ollama Llama-3.1...")
llm = Ollama(model="llama3.1", temperature=0.1, num_ctx=8000, verbose=False)

print("[INFO] Loading FlagReranker...")
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================
def is_global_summary_query(query: str) -> bool:
    trigger_words = ["list all figures", "all images", "summary of all visual", "what images are in"]
    return any(trigger in query.lower() for trigger in trigger_words)


def expand_query(original_query: str) -> str:
    try:
        chain = QUERY_EXPANSION_PROMPT | llm
        response = chain.invoke({"query": original_query})
        
        raw_text = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if not json_match:
            raise ValueError(f"No JSON object found in response: {raw_text}")
            
        clean_json_string = json_match.group(0)
        response_data = json.loads(clean_json_string)
        expanded = response_data.get("expanded_query", original_query)
        
        logging.info(f"[QUERY EXPANSION] Original: '{original_query}' -> Expanded: '{expanded}'")
        return expanded
        
    except Exception as e:
        logging.warning(f"Query expansion failed: {e}. Using original query.")
        return original_query


def format_references(safe_context_json: str):
    references = []
    try:
        context_data = json.loads(safe_context_json)
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse safe_context JSON for formatting references.")
        return []
    
    for doc in context_data:
        meta = doc.get("metadata", {})
        idx = doc.get("document_index")
        
        headers = [
            meta.get("Chapter", "").replace("**", ""),
            meta.get("Subsection", ""),
            meta.get("Subsubsection", ""),
            meta.get("Subsubsubsubection", "")
        ]
        breadcrumb = " > ".join([h for h in headers if h])
        
        source_path = meta.get("source", "")
        file_name = os.path.basename(source_path) if source_path else "Unknown Document"
        
        ref_entry = {
            "citation_index": idx,
            "display_title": file_name.replace("_hydrated.md", "").replace("_", " "),
            "breadcrumb": breadcrumb,
            "source_file": file_name,
            "figures": meta.get("has_figures", []),
            "tables": meta.get("has_tables", []),
            "content_snippet": doc.get("content", "")[:300] + "...", 
            "lineage": {
                "parent_id": meta.get("parent_id", meta.get("chunk_id")),
                "parent_hash": meta.get("parent_hash", ""),
                "version_hash": meta.get("version_hash", ""),
                "last_updated": meta.get("last_updated", "")
            }
        }
        references.append(ref_entry)
        
    return references


# ==============================================================================
# DEBUG FUNCTION: Child-Level Search
# ==============================================================================
def child_reranking(user_query: str):
    """
    Standalone diagnostic function to test direct child-level retrieval 
    and BGE cross-encoder reranking. 
    """
    print("\n[DEBUG] Executing direct child vector search...")
    child_docs_retrieved = vector_store.similarity_search(
        user_query, 
        k=20,
        filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.content_tier",
                    match=rest.MatchValue(value="core_science")
                )
            ]
        )
    )

    if not child_docs_retrieved:
        print("[DEBUG] No child documents retrieved.")
        return None

    pairs = [[user_query, doc.page_content] for doc in child_docs_retrieved]
    scores = reranker.compute_score(pairs)
    if isinstance(scores, float):
        scores = [scores]

    scored_children = sorted(zip(child_docs_retrieved, scores), key=lambda x: x[1], reverse=True)

    ordered_parent_ids = []
    seen_ids = set()

    for child, score in scored_children:
        parent_id = child.metadata.get("parent_id") or child.metadata.get("chunk_id")
        if parent_id and parent_id not in seen_ids:
            seen_ids.add(parent_id)
            ordered_parent_ids.append(parent_id)

    parent_docs = []
    if ordered_parent_ids:
        # Fetch only the Top 3 unique parents to safely fit the context budget
        raw_parents = retriever.docstore.mget(ordered_parent_ids[:3])
        parent_docs = [p for p in raw_parents if p is not None]

    # Fallback to children if parent lookup fails
    base_docs_for_expansion = parent_docs if parent_docs else [doc for doc, score in scored_children]

    print(f"[DEBUG] Ordered Parent IDs from Children: {ordered_parent_ids}")
    print(f"[DEBUG] Unique Parent IDs count: {len(seen_ids)}\n")
    return child_docs_retrieved, ordered_parent_ids, base_docs_for_expansion


def parent_reranking(user_query: str):
    """
    Standalone diagnostic function to test direct parent-level retrieval 
    and BGE cross-encoder reranking. 
    """
    print("\n Executing direct parent vector search...")
    # -------------------------------------------------------------------------
    # STAGE 1: BROAD PARENT RETRIEVAL (WITH FILTER INJECTED)
    # -------------------------------------------------------------------------
    # Inject the filter so Qdrant ignores bibliographies/noise during the underlying child search
    retriever.search_kwargs = {
        "k": 20, 
        "filter": rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.content_tier",
                    match=rest.MatchValue(value="core_science")
                )
            ]
        )
    }

    # retriever.invoke returns the full Parent Documents
    retrieved_parents = retriever.invoke(user_query)

    if not retrieved_parents:
        return "I do not have enough information in the provided context to answer this question.", "", [], []

    # -------------------------------------------------------------------------
    # STAGE 2: CROSS-ENCODER RERANKING
    # -------------------------------------------------------------------------
    # Note: BGE Reranker will only score the first 512 tokens of these parent documents
    pairs = [[user_query, doc.page_content] for doc in retrieved_parents]
    scores = reranker.compute_score(pairs)
    
    if isinstance(scores, float):
        scores = [scores]

    scored_parents = sorted(zip(retrieved_parents, scores), key=lambda x: x[1], reverse=True)
    top_parents = [doc for doc, score in scored_parents[:3]]

    return retrieved_parents, scored_parents, top_parents


# ==============================================================================
# 3. MAIN INFERENCE PIPELINE
# ==============================================================================
def generate_context(user_query: str):
    """
    Production RAG Inference:
    1. Parent retrieval via filtered retriever.invoke
    2. Reranking fetched Parent docs
    3. Sibling chunk expansion & token budgeting
    """
    
    # -------------------------------------------------------------------------
    # Child Reranking
    # -------------------------------------------------------------------------
    retrieved_parents, ordered_parent_ids, base_docs_for_expansion = child_reranking(user_query)

    # -------------------------------------------------------------------------
    # Parent Reranking
    # -------------------------------------------------------------------------
    # retrieved_parents, scored_parents, top_parents = parent_reranking(user_query)

    # -------------------------------------------------------------------------
    # STAGE 3: FORWARD NEIGHBOR EXPANSION & SAFE CONTEXT BUDGETING
    # -------------------------------------------------------------------------
    expanded_docs = [
        expand_following_neighbors(doc, retriever=retriever, min_tokens=1000)
        for doc in base_docs_for_expansion
    ]

    # Expanded to 6000+ budget to accommodate the bulky parent docs and neighbors
    safe_context = build_safe_context(expanded_docs, max_tokens=5000)
    references = format_references(safe_context)

    return safe_context, expanded_docs, references, retrieved_parents


def generate_answer(user_query: str):
    
    # Optional: Enable Query Expansion if needed
    # optimized_query = expand_query(user_query)
    # safe_context, expanded_docs, references, retrieved_parents = generate_context(user_query=optimized_query)
    
    safe_context, expanded_docs, references, retrieved_parents = generate_context(user_query=user_query)

    formatted_prompt = STRICT_QA_PROMPT.format(
            context=safe_context,
            question=user_query
        )
    
    response = llm.invoke(formatted_prompt)
    answer_text = response.content if hasattr(response, "content") else str(response)

    print("\n[ANSWER]\n", answer_text)

    return answer_text, safe_context, expanded_docs, references, retrieved_parents


# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    test_query = "What is the likely range of the contribution of internal variability to global surface temperature warming between 2010 and 2019 relative to 1850\u20131900?"
    
    try:
        generate_answer(user_query=test_query)
        
    finally:
        print("\n[INFO] Closing database connections...")
        client.close()