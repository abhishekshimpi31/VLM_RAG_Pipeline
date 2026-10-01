import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.indexing.chunking_vectordb import process_and_ingest, vector_store

def main():
    print("[INFO] Starting IPCC document ingestion into Qdrant & Docstore...")
    process_and_ingest()
    vector_store.client.close()
    print("[INFO] Ingestion process finished successfully.")

if __name__ == "__main__":
    main()
