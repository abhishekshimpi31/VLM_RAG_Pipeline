import hashlib
import json
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Any

from langsmith import traceable

import config


def init_cache_collection(client, rest, collection_name: str, vector_size: int = 1024) -> None:
    """
    Idempotently creates the Qdrant semantic cache collection with on-disk payload storage
    to avoid allocating RAM on memory-constrained systems.
    """
    try:
        if not client.collection_exists(collection_name):
            logging.info(f"[CACHE] Creating Qdrant cache collection: '{collection_name}' (dim: {vector_size})")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
                on_disk_payload=True
            )
    except Exception as e:
        logging.error(f"[CACHE ERROR] Failed to initialize cache collection '{collection_name}': {e}")


def generate_query_id(text: str) -> str:
    """
    Generates a deterministic UUID from the normalized query string.
    Ensures repeated queries overwrite existing cache slots rather than creating duplicates.
    """
    hash_str = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
    return f"{hash_str[:8]}-{hash_str[8:12]}-{hash_str[12:16]}-{hash_str[16:20]}-{hash_str[20:]}"


@traceable(run_type="retriever", name="cache_lookup")
def get_cached_result(
    query: str,
    client,
    embedding_model,
    rest,
    collection_name: str,
    threshold: float = 0.96
) -> Optional[Tuple[str, str, List[dict], List[dict]]]:
    """
    Checks Qdrant for a semantically equivalent query. Returns deserialized 
    (answer_text, safe_context, expanded_docs, references) if similarity >= threshold.
    """
    try:
        # Lazy initialization
        # init_cache_collection(client=client, rest=rest, collection_name=config.CACHE_COLLECTION_NAME, vector_size=1024)

        query_vector = embedding_model.embed_query(query)

        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=1,
            with_payload=True
        )

        results = response.points

        if results and results[0].score >= threshold:
            logging.info(f"[CACHE HIT] Similarity: {results[0].score:.4f} for query: '{query}'")
            payload = results[0].payload

            answer_text = payload.get("answer_text", "")
            safe_context = payload.get("safe_context", "{}")
            expanded_docs = json.loads(payload.get("expanded_docs_json", "[]"))
            references = json.loads(payload.get("references_json", "[]"))

            return answer_text, safe_context, expanded_docs, references

        return None

    except Exception as e:
        logging.warning(f"[CACHE WARN] Lookup failed; bypassing cache. Error: {e}")
        return None


@traceable(run_type="tool", name="cache_store")
def set_cached_result(
    query: str,
    answer_text: str,
    safe_context: str,
    expanded_docs: list,
    references: list,
    client,
    rest,
    embedding_model,
    collection_name: str
) -> None:
    """
    Serializes and upserts the full context state and generated answer into Qdrant.
    """
    try:
        query_vector = embedding_model.embed_query(query)
        point_id = generate_query_id(query)

        simplified_docs = [
            {"page_content": d.page_content, "metadata": d.metadata}
            if hasattr(d, "page_content") else d
            for d in expanded_docs
        ]

        client.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={
                        "original_query": query,
                        "answer_text": answer_text,
                        "safe_context": safe_context,
                        "expanded_docs_json": json.dumps(simplified_docs),
                        "references_json": json.dumps(references),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )
        logging.info(f"[CACHE] Saved entry for query: '{query[:50]}...'")

    except Exception as e:
        logging.warning(f"[CACHE WARN] Failed to save entry to cache. Error: {e}")