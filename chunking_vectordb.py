import glob
import os
from datetime import datetime  # Fixed import
import logging

from dotenv import load_dotenv
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client import models

# LangChain Imports
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore

# Your custom module
from file_versioning import get_file_hash 

# -------------------------------------------------------------
# 0. System Configuration
# -------------------------------------------------------------
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

QDRANT_DB_PATH = "vector_database/quadrant_database/qdrant_db"
DOCSTORE_PATH = "vector_database/quadrant_database/docstore" # Moved outside of qdrant_db to prevent file lock issues

# -------------------------------------------------------------
# 1. Initialize Models (With Production Normalization)
# -------------------------------------------------------------
logging.info("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': True}  # CRITICAL: Required for accurate Cosine distance
)

# -------------------------------------------------------------
# 2. Setup Qdrant & Disk-Tiering (Hardware Optimization)
# -------------------------------------------------------------
logging.info("Setting up Qdrant and Parent Document Store...")
client = QdrantClient(path=QDRANT_DB_PATH)
collection_name = "ipcc_child_chunks"

# Only create the collection if it doesn't exist. This allows us to run 
# the script incrementally day after day without wiping the whole DB!
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        # --- PRODUCTION OPTIMIZATIONS ---
        on_disk_payload=True,  # Moves heavy metadata to SSD
        hnsw_config=models.HnswConfigDiff(on_disk=True),  # Moves vector index to SSD
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2)
    )

vector_store = QdrantVectorStore(
    client=client, 
    collection_name=collection_name, 
    embedding=embedding_model
)

store = LocalFileStore(DOCSTORE_PATH)

# -------------------------------------------------------------
# 3. Setup the AST Splitters
# -------------------------------------------------------------
headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
]
# CRITICAL: strip_headers MUST be False so the LLM can read the section titles
parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

def token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,          
    chunk_overlap=50,
    length_function=token_length,
    separators=["\n\n", "\n> ", "\n", " ", ""],
    keep_separator=True, 
    add_start_index=True,  
    strip_whitespace=True 
)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    byte_store=store,
    child_splitter=child_splitter,
    search_type="mmr",               
    search_kwargs={
        "k": 5,                      
        "fetch_k": 20,
        "score_threshold": 0.50              
    }
)

# -------------------------------------------------------------
# 4. Ingestion & Version Syncing
# -------------------------------------------------------------
folders = glob.glob("data/extracted_data/**/md/")
logging.info(f"Found {len(folders)} markdown folders for ingestion.")

all_parent_docs = []

for folder in folders:
    doc_type = os.path.basename(os.path.normpath(folder))
    loader = DirectoryLoader(folder, glob="*_hydrated.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    folder_docs = loader.load()
    
    for doc in folder_docs:
        filepath = doc.metadata['source']
        current_hash = get_file_hash(filepath)
        modified_timestamp = os.path.getmtime(filepath)
        formatted_date = datetime.fromtimestamp(modified_timestamp).strftime('%Y-%m-%d')
    
        logging.info(f"Processing {filepath} | Version Hash: {current_hash[:8]}")

        # Delete OLD vectors for this specific file before adding new ones
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",  # LangChain nests metadata under the 'metadata' key [1.1.3]
                            match=models.MatchValue(value=filepath)
                        )
                    ]
                )
            )
        )
        
        # Split text into AST parent sections
        ast_sections = parent_splitter.split_text(doc.page_content)
        
        for section in ast_sections:
            section.metadata["doc_type"] = doc_type
            section.metadata["source"] = filepath
            section.metadata["version_hash"] = current_hash
            section.metadata["last_updated"] = formatted_date

            all_parent_docs.append(section)

# -------------------------------------------------------------
# 5. Batched Vectorization (Prevents RAM crashes)
# -------------------------------------------------------------
if all_parent_docs:
    BATCH_SIZE = 150
    logging.info(f"Extracted {len(all_parent_docs)} Parent Sections. Starting Batched Ingestion...")
    
    for i in range(0, len(all_parent_docs), BATCH_SIZE):
        batch = all_parent_docs[i : i + BATCH_SIZE]
        retriever.add_documents(batch, ids=None)
        logging.info(f" -> Ingested batch {i//BATCH_SIZE + 1}/{(len(all_parent_docs)//BATCH_SIZE) + 1}")

    logging.info(f"Ingestion complete!")
    logging.info(f" -> Child Vectors searchable in Qdrant: {client.count(collection_name).count}")
    vector_store.client.close()
else:
    logging.warning("No documents found to ingest!")
    