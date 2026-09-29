import json
import math
import pandas as pd
from tqdm import tqdm

# Import only your retrieval components to bypass the slow LLM generation step
from inference import  generate_context

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BENCHMARK_PATH = "data/extracted_data/testing_data/retrieval_benchmark.json"
OUTPUT_CSV = "data/extracted_data/testing_data/retrieval_metrics.csv"

# ==============================================================================
# 2. DETERMINISTIC IR METRICS
# ==============================================================================
def compute_ir_metrics(relevance: list[int]) -> dict:
    """Computes Precision@K, Recall@K, MRR@K, and NDCG@K from binary hits [1, 0, ...]."""
    k = len(relevance)
    if k == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}

    precision_k = sum(relevance) / k
    recall_k = 1.0 if sum(relevance) > 0 else 0.0

    mrr_k = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel == 1:
            mrr_k = 1.0 / rank
            break

    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    ndcg_k = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "precision_at_k": round(precision_k, 4),
        "recall_at_k": round(recall_k, 4),
        "mrr_at_k": round(mrr_k, 4),
        "ndcg_at_k": round(ndcg_k, 4),
    }

import json

def get_unique_parent_ids(safe_context_json: str) -> list:
    """
    Parses the safe_context JSON string and returns a ranked list 
    of unique parent_ids that survived the token budget.
    """
    try:
        context_data = json.loads(safe_context_json)
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse safe_context JSON.")
        return []
        
    unique_parents = []
    
    for doc in context_data:
        metadata = doc.get("metadata", {})
        # Grab parent_id (fallback to chunk_id if parent_id is missing)
        parent_id = metadata.get("parent_id") or metadata.get("chunk_id")
        
        if parent_id and parent_id not in unique_parents:
            unique_parents.append(parent_id)
            
    return unique_parents

# ==========================================
# Example Usage in your inference loop:
# ==========================================
# safe_context, expanded_docs, references = generate_context(q)
# surviving_parents = get_unique_parent_ids(safe_context)
# print(f"Unique Parents passed to LLM: {surviving_parents}")

def analyze_retrieved_chunks(docs):
    """
    Analyzes a list of LangChain Document objects to count total chunks
    and unique parent IDs.
    """
    unique_parents = set()
    total_chunks = len(docs)

    for doc in docs:
        # Works for both raw Document objects or dictionaries if you parsed the JSON
        meta = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
        
        p_id = meta.get('parent_id') or meta.get('chunk_id')
        if p_id:
            unique_parents.add(p_id)

    # Return the dictionary so you can print it or use the numbers
    return total_chunks, len(unique_parents), list(unique_parents)
    

# --- How to use it ---
# If you want to see the raw child chunks from Qdrant:
# raw_child_docs = vector_store.similarity_search(q, k=20)
# analyze_retrieved_chunks(raw_child_docs)  

# ==============================================================================
# 3. PIPELINE EXECUTION
# ==============================================================================
def run_evaluation():
    print(f"\n[1/3] Loading Benchmark: '{BENCHMARK_PATH}'...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_records = json.load(f)

    eval_data = []

    print(f"[2/3] Running {len(benchmark_records)} queries through retrieval engine...")
    for item in tqdm(benchmark_records, desc="Evaluating"):
        q = item["question"]
        target_id = item["target_parent_id"]

        print("target id", target_id)

        # 1. Execute purely the vector search
        safe_context, expanded_docs, references, child_docs = generate_context(q)

        total_chunks, unique_parents_count, unique_parents = analyze_retrieved_chunks(child_docs)

        print("Parent id from child docs", unique_parents)
        print("Unique Parent id count from child docs", unique_parents)

        surviving_parents = get_unique_parent_ids(safe_context)

        print(f"Unique Parents passed to LLM: {surviving_parents}")

        # 2. Parse the JSON string back into a list of dictionaries
        try:
            context_data = json.loads(safe_context)
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse safe_context for query: {q}")
            context_data = []

        # 3. Build the binary relevance array [0, 1, 0, ...]
        relevance_list = []
        
        # Iterate over the parsed dictionaries
        for doc in context_data:
            # Use dictionary `.get()` since these are no longer LangChain objects
            metadata = doc.get("metadata", {})
            # print("metadata", metadata)
            doc_id = metadata.get("parent_id") or metadata.get("chunk_id")

            # print(f"Checking target id {target_id} with doc id {doc_id}")
            
            if doc_id == target_id:
                relevance_list.append(1)
            else:
                relevance_list.append(0)

        print("relevance_list", relevance_list)
        # 4. Compute deterministic metrics for this query
        ir_scores = compute_ir_metrics(relevance_list)

        eval_data.append({
            "question": q,
            "target_parent_id": target_id,
            "num_chunks_retrieved": len(context_data), # Count what actually survived truncation
            "relevance_vector": str(relevance_list),
            **ir_scores
        })

    # 4. Process and Display Results
    df_results = pd.DataFrame(eval_data)
    
    print("\n" + "=" * 60)
    print("                RETRIEVAL BENCHMARK SUMMARY                 ")
    print("=" * 60)
    print(f"Total Queries Evaluated : {len(df_results)}")
    print(f"Precision@K             : {df_results['precision_at_k'].mean():.4f}")
    print(f"Recall@K (HitRate)      : {df_results['recall_at_k'].mean():.4f}")
    print(f"MRR@K                   : {df_results['mrr_at_k'].mean():.4f}")
    print(f"NDCG@K                  : {df_results['ndcg_at_k'].mean():.4f}")
    print("=" * 60)

    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Detailed evaluation results saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    run_evaluation()