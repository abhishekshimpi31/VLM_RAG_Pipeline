import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.evaluate_retrieval import run_evaluation

def main():
    print("[INFO] Starting retrieval benchmark evaluation (MRR, NDCG, HitRate, Precision)...")
    run_evaluation()

if __name__ == "__main__":
    main()
