import sys
import types
import json
import math
import re
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# 0. HOTFIX FOR RAGAS DEPENDENCY BUG
# ==============================================================================
if 'langchain_community.chat_models' not in sys.modules:
    sys.modules['langchain_community.chat_models'] = types.ModuleType('langchain_community.chat_models')

fake_vertexai = types.ModuleType('langchain_community.chat_models.vertexai')
fake_vertexai.ChatVertexAI = type('ChatVertexAI', (object,), {})
fake_vertexai.VertexAI = type('VertexAI', (object,), {})
sys.modules['langchain_community.chat_models.vertexai'] = fake_vertexai

# LangChain & Ragas imports
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Production inference & vector store components
from inference import generate_answer
from chunking_vectordb import embedding_model

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
GROUND_TRUTH_PATH = "data/extracted_data/testing_data/ground_truth_dataset.json"
OUTPUT_CSV = "data/extracted_data/testing_data/evaluation_results.csv"
MAX_TEST_SAMPLES = None  # Set to an integer (e.g. 18) or None to evaluate all

# Judge models
llm = ChatOllama(model="llama3", temperature=0.0, num_ctx=6000, format="json")
ragas_llm = LangchainLLMWrapper(llm)
ragas_embed = LangchainEmbeddingsWrapper(embedding_model)

# ==============================================================================
# 2. DETERMINISTIC IR METRICS (Chunk Evaluation)
# ==============================================================================
def determine_relevance(safe_context_list: list[str], target_id: str | None, reference_context: str, docs: list) -> list[int]:
    """
    Evaluates relevance for each retrieved chunk in safe_context_list:
    - Primary: Exact chunk_id match from retrieved doc metadata.
    - Fallback: Substantial textual containment / token overlap against reference_context.
    """
    relevance = []
    ref_clean = " ".join(reference_context.lower().split()) if reference_context else ""
    ref_tokens = set(ref_clean.split())

    for idx, chunk_text in enumerate(safe_context_list):
        is_rel = 0

        # 1. Exact ID Matching (checked via corresponding retrieved Document object)
        if target_id and idx < len(docs):
            doc_chunk_id = docs[idx].metadata.get("chunk_id")
            if doc_chunk_id and doc_chunk_id == target_id:
                is_rel = 1

        # 2. Textual Containment Fallback (if ID missing or did not match)
        if is_rel == 0 and ref_tokens:
            chunk_clean = " ".join(chunk_text.lower().split())
            if (ref_clean[:120] and ref_clean[:120] in chunk_clean) or (chunk_clean[:120] and chunk_clean[:120] in ref_clean):
                is_rel = 1
            else:
                chunk_tokens = set(chunk_clean.split())
                overlap = len(chunk_tokens & ref_tokens) / len(ref_tokens) if ref_tokens else 0.0
                if overlap >= 0.35:
                    is_rel = 1

        relevance.append(is_rel)

    return relevance


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

# ==============================================================================
# 3. PIPELINE EXECUTION & EVALUATION
# ==============================================================================
def run_evaluation():
    print(f"\n[1/3] Loading Ground Truth: '{GROUND_TRUTH_PATH}'...")
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth_records = json.load(f)

    if MAX_TEST_SAMPLES:
        ground_truth_records = ground_truth_records[:MAX_TEST_SAMPLES]

    eval_data = []
    ragas_samples = []

    print(f"[2/3] Running {len(ground_truth_records)} queries through inference pipeline...")
    for item in tqdm(ground_truth_records, desc="Evaluating"):
        q = item["question"]
        truth = item["ground_truth"]
        ref_context = item.get("reference_context", "")
        target_id = item.get("target_chunk_id") or item.get("metadata", {}).get("chunk_id")

        # Execute generation pipeline
        answer, safe_context, docs = generate_answer(q)

        # Parse context fed to the generator
        safe_context_split = re.split(r'--- Document \d+ ---', safe_context)
        safe_context_list = [c.strip() for c in safe_context_split if c.strip()]

        # 1. Deterministic IR Metrics
        relevance_list = determine_relevance(safe_context_list, target_id, ref_context, docs)
        ir_scores = compute_ir_metrics(relevance_list)

        eval_data.append({
            "question": q,
            "ground_truth": truth,
            "answer": answer,
            "num_chunks_retrieved": len(safe_context_list),
            "relevance_vector": str(relevance_list),
            **ir_scores
        })

        # 2. Package for Ragas evaluation
        ragas_samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer if answer else "No answer generated.",
                retrieved_contexts=safe_context_list,
                reference=truth
            )
        )

    df_results = pd.DataFrame(eval_data)
    eval_dataset = EvaluationDataset(ragas_samples)

    # 3. LLM-as-a-Judge Evaluation (Ragas)
    print("\n[3/3] Running Ragas LLM-as-a-Judge...")

    custom_run_config = RunConfig(
    timeout=600,       # Wait up to 10 minutes for Ollama to respond
    max_workers=1,     # Do not send concurrent requests
    max_retries=2      # Retry if a request fails
)

    judge_results = evaluate(
        dataset=eval_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embedding_model,
        raise_exceptions=True,
        run_config=custom_run_config
    )

    df_judge = judge_results.to_pandas()
    for metric_col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        df_results[metric_col] = df_judge[metric_col]

    # Display Metrics Summary
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
    print(f"Context Recall   : {df_results['context_recall'].mean():.4f}")
    print(f"Context Precision: {df_results['context_precision'].mean():.4f}")
    print("=" * 60)

    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SUCCESS] Detailed evaluation results saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    run_evaluation()