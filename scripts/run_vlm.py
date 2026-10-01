import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from src.ingestion.vlm_model import initialize_model, run_vlm_router

def main():
    parser = argparse.ArgumentParser(description="Run Qwen2-VL figure extraction on pending manifests")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for image processing")
    parser.add_argument("--base-dir", type=str, default=str(getattr(config, "EXTRACTED_DATA_DIR", "data/extracted_data/")), help="Extracted data directory")
    args = parser.parse_args()

    print("[INFO] Initializing VLM Model...")
    model, processor = initialize_model()
    print(f"[INFO] Running VLM router on directory: {args.base_dir}")
    run_vlm_router(model=model, processor=processor, base_dir=args.base_dir, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
