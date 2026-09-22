import sys
import types
import os
import re
import json
import math
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

# ==============================================================================
# 0. HOTFIX FOR RAGAS DEPENDENCY BUG
# Fakes the missing Google VertexAI module so Ragas imports without error
# ==============================================================================
if 'langchain_community.chat_models' not in sys.modules:
    sys.modules['langchain_community.chat_models'] = types.ModuleType('langchain_community.chat_models')

fake_vertexai = types.ModuleType('langchain_community.chat_models.vertexai')
fake_vertexai.ChatVertexAI = type('ChatVertexAI', (object,), {})
fake_vertexai.VertexAI = type('VertexAI', (object,), {})
sys.modules['langchain_community.chat_models.vertexai'] = fake_vertexai

# LangChain & Ragas imports
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Direct import of your production inference pipeline
from inference import generate_answer

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
GROUND_TRUTH_PATH = "data/extracted_data/testing_data/ground_truth_dataset.json"
MAX_TEST_SAMPLES = 50  # Number of samples to evaluate
OUTPUT_CSV = "data/extracted_data/testing_data/evaluation_results.csv"

# ==============================================================================
# 2. DETERMINISTIC IR METRICS (Chunk Evaluation)
# ==============================================================================
def determine_relevance(docs: list, target_id: str, reference_context: str) -> list[int]:
    """
    Evaluates relevance for each retrieved chunk:
    - Primary: Exact chunk_id match if present in metadata.
    - Fallback: Substantial textual containment against reference_context.
    """
    relevance = []
    ref_clean = " ".join(reference_context.lower().split()) if reference_context else ""
    ref_tokens = set(ref_clean.split())

    for doc in docs:
        is_rel = 0
        doc_chunk_id = doc.metadata.get("chunk_id")

        # 1. Exact ID Matching
        if target_id and doc_chunk_id and target_id == doc_chunk_id:
            is_rel = 1
        # 2. Textual Containment Fallback (if JSON lacks chunk_id)
        elif ref_tokens:
            doc_clean = " ".join(doc.page_content.lower().split())
            if ref_clean[:120] in doc_clean or doc_clean[:120] in ref_clean:
                is_rel = 1
            else:
                doc_tokens = set(doc_clean.split())
                overlap = len(doc_tokens & ref_tokens) / len(ref_tokens)
                if overlap >= 0.35:
                    is_rel = 1

        relevance.append(is_rel)
    return relevance

def compute_ir_metrics_from_relevance(relevance: list[int]) -> dict:
    """Computes Precision@K, Recall@K, MRR@K, and NDCG@K from binary hits [1, 0, ...]."""
    k = len(relevance)
    if k == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}

    # Precision@K: Fraction of retrieved chunks that are relevant
    precision_k = sum(relevance) / k

    # Recall@K (Hit Rate): 1 if at least one relevant chunk was retrieved
    recall_k = 1.0 if sum(relevance) > 0 else 0.0

    # MRR@K: Reciprocal rank of the first relevant chunk
    mrr_k = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel == 1:
            mrr_k = 1.0 / rank
            break

    # NDCG@K: Logarithmically discounted ranking metric
    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    ndcg_k = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "precision_at_k": round(precision_k, 4),
        "recall_at_k": round(recall_k, 4),
        "mrr_at_k": round(mrr_k, 4),
        "ndcg_at_k": round(ndcg_k, 4)
    }

# ==============================================================================
# 3. LLM-AS-A-JUDGE CONFIGURATION
# ==============================================================================
print("[INFO] Initializing Ragas Judge Models...")
judge_llm = ChatOllama(model="llama3", temperature=0.0, num_ctx=6000, verbose=False)
ragas_llm = LangchainLLMWrapper(judge_llm)

judge_embed = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
ragas_embed = LangchainEmbeddingsWrapper(judge_embed)

for metric in [faithfulness, answer_relevancy]:
    metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embed

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
def run_evaluation():
    print(f"\n[1/3] Loading Ground Truth: '{GROUND_TRUTH_PATH}'...")
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth_records = json.load(f)[:MAX_TEST_SAMPLES]

    eval_data = []
    ragas_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"[2/3] Running {len(ground_truth_records)} queries through inference pipeline...")

    for item in tqdm(ground_truth_records, desc="Evaluating"):
        q = item["question"]
        truth = item["ground_truth"]
        ref_context = item.get("reference_context", "")
        target_id = item.get("metadata", {}).get("chunk_id")

        # Single execution: answers, context string, and structured documents
        answer, safe_context, docs = generate_answer(q)

        # Extract text contents directly from the returned Document objects
        retrieved_contexts = [doc.page_content for doc in docs] if docs else ["No context found."]

        # Calculate deterministic IR metrics on the retrieved chunks
        relevance_list = determine_relevance(docs, target_id, ref_context)
        ir_scores = compute_ir_metrics_from_relevance(relevance_list)

        eval_data.append({
            "question": q,
            "ground_truth": truth,
            "answer": answer,
            "num_chunks_retrieved": len(docs),
            "relevance_vector": str(relevance_list),
            **ir_scores
        })

        # Append to Ragas payload
        ragas_dict["question"].append(q)
        ragas_dict["answer"].append(answer if answer else "No answer generated.")
        ragas_dict["contexts"].append(retrieved_contexts)
        ragas_dict["ground_truth"].append(truth)

    df_results = pd.DataFrame(eval_data)

    # Step 3: Run LLM-as-a-Judge for Generation Metrics
    print("\n[3/3] Running Ragas LLM-as-a-Judge (Faithfulness & Answer Relevancy)...")
    ragas_dataset = Dataset.from_dict(ragas_dict)

    judge_results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        raise_exceptions=False
    )

    df_judge = judge_results.to_pandas()
    df_results["faithfulness"] = df_judge["faithfulness"]
    df_results["answer_relevancy"] = df_judge["answer_relevancy"]

    # Summary Display
    print("\n" + "=" * 60)
    print("                 PIPELINE EVALUATION SUMMARY                 ")
    print("=" * 60)
    print("--- RETRIEVAL METRICS (CHUNKS) ---")
    print(f"Precision@K      : {df_results['precision_at_k'].mean():.4f}")
    print(f"Recall@K (HitRate): {df_results['recall_at_k'].mean():.4f}")
    print(f"MRR@K            : {df_results['mrr_at_k'].mean():.4f}")
    print(f"NDCG@K           : {df_results['ndcg_at_k'].mean():.4f}")
    print("\n--- GENERATION METRICS (LLM-AS-A-JUDGE) ---")
    print(f"Faithfulness     : {df_results['faithfulness'].mean():.4f}")
    print(f"Answer Relevancy : {df_results['answer_relevancy'].mean():.4f}")
    print("=" * 60)

    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Detailed evaluation results saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    run_evaluation()