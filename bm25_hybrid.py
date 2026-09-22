import os
import pickle
import logging
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Import existing Qdrant objects 
from chunking_vectordb import client, collection_name, vector_store

BM25_INDEX_PATH = "vector_database/quadrant_database/bm25_index.pkl"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clean_bm25_tokenizer(text: str) -> list[str]:
    # Strip markdown syntax and punctuation, keeping alphanumeric strings, decimals, and hyphens
    cleaned = re.sub(r'[\*#_\[\]\(\),;:\"]', ' ', text.lower())
    return [token for token in cleaned.split() if len(token) > 1]


def build_and_save_bm25_index(output_path: str = BM25_INDEX_PATH, k: int = 8) -> BM25Retriever:
    """
    Extracts child chunks directly from the running local Qdrant collection,
    fits a BM25 sparse index, and pickles it to disk.
    
    Zero embeddings are computed; no vectors are re-indexed.
    """
    logging.info(f"Extracting child chunks from Qdrant collection '{collection_name}'...")
    
    child_documents = []
    next_offset = None
    
    # Scroll through Qdrant collection to retrieve payloads without vectors
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=next_offset
        )
        
        for record in records:
            payload = record.payload or {}
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            if content:
                child_documents.append(Document(page_content=content, metadata=metadata))
                
        if next_offset is None:
            break

    if not child_documents:
        raise ValueError(
            f"No points found in Qdrant collection '{collection_name}'. "
            "Ensure ingestion has run at least once."
        )

    logging.info(f"Fitting BM25 index on {len(child_documents)} child chunks...")
    bm25_retriever = BM25Retriever.from_documents(child_documents, preprocess_func=clean_bm25_tokenizer)
    bm25_retriever.k = k

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    logging.info(f"[SUCCESS] BM25 index saved to '{output_path}'")
    return bm25_retriever


def load_bm25_retriever(index_path: str = BM25_INDEX_PATH, k: int = 8) -> BM25Retriever:
    """Loads the pickled BM25Retriever from disk, or builds it if absent."""
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            bm25 = pickle.load(f)
            bm25.k = k
            return bm25
    
    logging.warning(f"Index not found at '{index_path}'. Generating it now...")
    return build_and_save_bm25_index(output_path=index_path, k=k)


def get_hybrid_retriever(k: int = 8, vector_weight: float = 0.80, bm25_weight: float = 0.20) -> EnsembleRetriever:
    """
    Fuses Qdrant Dense Vector search with BM25 Sparse Keyword search
    using Reciprocal Rank Fusion (RRF).
    """
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    sparse_retriever = load_bm25_retriever(k=k)
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[vector_weight, bm25_weight]
    )
    return hybrid_retriever


if __name__ == "__main__":
    # Run once directly to construct and persist the BM25 index
    build_and_save_bm25_index(k=8)
    vector_store.client.close()