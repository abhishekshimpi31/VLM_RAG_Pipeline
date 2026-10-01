import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from src.ingestion.image_context_ingestion import hydrate_markdown_files

def main():
    base_dir = str(getattr(config, "EXTRACTED_DATA_DIR", "data/extracted_data/"))
    print(f"[INFO] Starting Markdown hydration from: {base_dir}")
    hydrate_markdown_files(base_dir=base_dir)
    print("[INFO] Markdown hydration complete.")

if __name__ == "__main__":
    main()
