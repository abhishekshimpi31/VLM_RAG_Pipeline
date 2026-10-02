import logging
import os
import re
import json
from typing import Optional, List, Dict, Any
import time

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langsmith import traceable

from FlagEmbedding import FlagReranker
from qdrant_client.http import models as rest

from src.common.model_dependency import get_embedding_model, get_llm, get_reranker
from src.retrieval.semantic_cache import get_cached_result, init_cache_collection, set_cached_result

from ..indexing.chunking_vectordb import build_safe_context, retriever, client, vector_store
from ..indexing.dynamic_splitter import expand_following_neighbors
import config


# ==============================================================================
# PROMPTS (Imported from central config)
# ==============================================================================
QUERY_EXPANSION_PROMPT_TEMPLATE = config.QUERY_EXPANSION_PROMPT_TEMPLATE
QA_PROMPT_TEMPLATE = config.QA_PROMPT_TEMPLATE
PROMPT_TEMPLATE_TEXT = QA_PROMPT_TEMPLATE

QUERY_EXPANSION_PROMPT = PromptTemplate.from_template(QUERY_EXPANSION_PROMPT_TEMPLATE)

EMBEDDING_MODEL_NAME = getattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

CACHE_COLLECTION_NAME = config.CACHE_COLLECTION_NAME

STRICT_QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

CONVERSATIONAL_QA_PROMPT_TEMPLATE = getattr(config, "CONVERSATIONAL_QA_PROMPT_TEMPLATE", QA_PROMPT_TEMPLATE)
CONVERSATIONAL_QA_PROMPT = PromptTemplate(
    template=CONVERSATIONAL_QA_PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)

CONDENSE_QUESTION_PROMPT_TEMPLATE = getattr(config, "CONDENSE_QUESTION_PROMPT_TEMPLATE", "")
if CONDENSE_QUESTION_PROMPT_TEMPLATE:
    CONDENSE_QUESTION_PROMPT = PromptTemplate(
        template=CONDENSE_QUESTION_PROMPT_TEMPLATE,
        input_variables=["chat_history", "question"]
    )
else:
    CONDENSE_QUESTION_PROMPT = None


# ==============================================================================
# 1. LLM & RERANKER SETUP
# ==============================================================================
llm = get_llm()
reranker = get_reranker()
embedding_model = get_embedding_model()


def format_chat_history(chat_history: list) -> str:
    """Formats message history [{'role': 'user'|'assistant', 'content': '...'}] into dialogue string."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history[-6:]:  # Keep up to last 3 conversation turns
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================
def is_global_summary_query(query: str) -> bool:
    trigger_words = ["list all figures", "all images", "summary of all visual", "what images are in"]
    return any(trigger in query.lower() for trigger in trigger_words)


@traceable(run_type="chain", name="Query Expansion")
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
# Child-Level Reranking
# ==============================================================================
@traceable(run_type="chain", name="bge_cross_encoder_child_rerank")
def child_reranking(user_query: str):
    """
    Standalone diagnostic function to test direct child-level retrieval 
    and BGE cross-encoder reranking. 
    """
    start_seq = time.perf_counter()

    print("\n[DEBUG] Executing direct child vector search...")
    child_docs_retrieved = vector_store.similarity_search(
        user_query, 
        k=getattr(config, "RETRIEVAL_TOP_K", 20),
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
        return None, [], []

    pairs = [[user_query, doc.page_content] for doc in child_docs_retrieved]
    scores = reranker.compute_score(pairs, batch_size=1)
    if isinstance(scores, float):
        scores = [scores]

    scored_children = sorted(zip(child_docs_retrieved, scores), key=lambda x: x[1], reverse=True)

    seq_duration = (time.perf_counter() - start_seq) * 1000  # Convert to ms
    print("seq_duration", seq_duration)

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
        raw_parents = retriever.docstore.mget(ordered_parent_ids[:getattr(config, "RERANKER_TOP_K", 3)])
        parent_docs = [p for p in raw_parents if p is not None]

    # Fallback to children if parent lookup fails
    base_docs_for_expansion = parent_docs if parent_docs else [doc for doc, score in scored_children]

    print(f"[DEBUG] Ordered Parent IDs from Children: {ordered_parent_ids}")
    print(f"[DEBUG] Unique Parent IDs count: {len(seen_ids)}\n")
    return child_docs_retrieved, ordered_parent_ids, base_docs_for_expansion


@traceable(run_type="chain", name="bge_cross_encoder_parent_rerank")
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
        "k": getattr(config, "RETRIEVAL_TOP_K", 20), 
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
    top_parents = [doc for doc, score in scored_parents[:getattr(config, "RERANKER_TOP_K", 3)]]

    return retrieved_parents, scored_parents, top_parents


# ==============================================================================
# 3. MAIN INFERENCE PIPELINE
# ==============================================================================
@traceable(run_type="retriever", name="generate_retrieval_context")
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
        expand_following_neighbors(doc, retriever=retriever, min_tokens=getattr(config, "EXPANSION_MIN_TOKENS", 1000))
        for doc in base_docs_for_expansion
    ]

    # Expanded budget to accommodate the bulky parent docs and neighbors
    safe_context = build_safe_context(expanded_docs, max_tokens=getattr(config, "MAX_SAFE_CONTEXT_TOKENS", 5000))
    references = format_references(safe_context)

    return safe_context, expanded_docs, references, retrieved_parents


@traceable(run_type="chain", name="generate_answer")
def generate_answer(user_query: str, chat_history: Optional[list] = None):
    """
    Executes conversational RAG pipeline:
    1. If chat history exists, optionally condenses follow-up into a standalone retrieval query.
    2. Retrieves parent documents via Qdrant & cross-encoder reranking.
    3. Synthesizes an answer strictly grounded in context with [n] citations.
    """
    retrieval_query = user_query
    formatted_history = format_chat_history(chat_history) if chat_history else ""

    # Condense follow-up questions for accurate vector retrieval if chat history exists
    if formatted_history and CONDENSE_QUESTION_PROMPT is not None:
        try:
            condense_chain = CONDENSE_QUESTION_PROMPT | llm
            condensed = condense_chain.invoke({
                "chat_history": formatted_history,
                "question": user_query
            })
            condensed_text = (condensed.content if hasattr(condensed, "content") else str(condensed)).strip()
            if condensed_text and len(condensed_text) > 4 and not condensed_text.startswith("Error"):
                retrieval_query = condensed_text
                print(f"[CONVERSATIONAL RAG] Reformulated search query: {retrieval_query}")
        except Exception as e:
            print(f"[WARNING] Conversational query condensation skipped: {e}")
            retrieval_query = user_query

    # Checking answer in the cache first
    cached_result = get_cached_result(
            query=retrieval_query,
            client=client,
            rest=rest,
            embedding_model=embedding_model,
            collection_name=config.CACHE_COLLECTION_NAME
        )
    
    if cached_result:
        answer_text, safe_context, expanded_docs, references =  cached_result
        print(f"\n[CACHE HIT - ~15ms]\n{answer_text}")
        return answer_text, safe_context, expanded_docs, references, []
    
    print("\n[CACHE MISS] Executing Child-Rerank-Parent pipeline...")
    safe_context, expanded_docs, references, retrieved_parents = generate_context(user_query=retrieval_query)

    if formatted_history:
        formatted_prompt = CONVERSATIONAL_QA_PROMPT.format(
            context=safe_context,
            chat_history=formatted_history,
            question=user_query
        )
    else:
        formatted_prompt = STRICT_QA_PROMPT.format(
            context=safe_context,
            question=user_query
        )
    
    response = llm.invoke(formatted_prompt)
    answer_text = response.content if hasattr(response, "content") else str(response)

    print("\n[ANSWER]\n", answer_text)

    # Saving the result to cache for future queries
    set_cached_result(
        query=user_query,
        answer_text=answer_text,
        safe_context=safe_context,
        expanded_docs=expanded_docs,
        references=references,
        client=client,
        rest=rest,
        embedding_model=embedding_model,
        collection_name=config.CACHE_COLLECTION_NAME
    )

    return answer_text, safe_context, expanded_docs, references, retrieved_parents



# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    test_query = "What is the best estimate of the anthropogenic attributable warming rate in degrees Celsius per decade for the period 2010\u20132019?"    
    try:
        generate_answer(user_query=test_query)
        
    finally:
        print("\n[INFO] Closing database connections...")
        client.close()
