import os
import sys
import argparse
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.inference import generate_answer, client

def main():
    parser = argparse.ArgumentParser(description="Run QA inference with IPCC Scientific RAG Pipeline")
    parser.add_argument(
        "--query", 
        type=str, 
        default="What is the best estimate of the anthropogenic attributable warming rate in degrees Celsius per decade for the period 2010\u20132019?",
        help="Scientific question to query the pipeline with"
    )
    args = parser.parse_args()

    print(f"\n[QUERY]: {args.query}\n")
    try:
        answer, safe_context, docs, references, retrieved = generate_answer(user_query=args.query)
        print("\n--- REFERENCES ---")
        for ref in references:
            print(f"[{ref['citation_index']}] {ref['display_title']} ({ref['breadcrumb']})")
    finally:
        print("\n[INFO] Closing database connections...")
        client.close()

if __name__ == "__main__":
    main()
