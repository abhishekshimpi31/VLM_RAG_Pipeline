import glob
import os
import hashlib
from datetime import datetime
import logging
import json
import re

from dotenv import load_dotenv
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client import models

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore

from src.common.model_dependency import get_embedding_model, get_embedding_tokenizer, get_llm_tokenizer, get_sparse_embedding_model


from .dynamic_splitter import token_length
from ..common.file_versioning import get_file_hash 
import config


# -------------------------------------------------------------
# 0. System Configuration
# -------------------------------------------------------------
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

QDRANT_DB_PATH = getattr(config, "QDRANT_DB_PATH", "vector_database/quadrant_database/qdrant_db")
DOCSTORE_PATH = getattr(config, "DOCSTORE_PATH", "vector_database/quadrant_database/docstore")
COLLECTION_NAME = getattr(config, "COLLECTION_NAME", "ipcc_hybrid_chunks")
EMBEDDING_MODEL_NAME = getattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
LLM_TOKENIZER_NAME = getattr(config, "LLM_TOKENIZER_NAME", "hf-internal-testing/llama-tokenizer")

tokenizer = get_embedding_tokenizer()
llm_tokenizer = get_llm_tokenizer()

# -------------------------------------------------------------
# 1. Initialize Models (Via Centralized Models Module)
# -------------------------------------------------------------
embedding_model = get_embedding_model()
sparse_embeddings = get_sparse_embedding_model()

# -------------------------------------------------------------
# 2. Setup Qdrant & Disk-Tiering
# -------------------------------------------------------------
logging.info("Setting up Qdrant and Parent Document Store...")
os.makedirs(QDRANT_DB_PATH, exist_ok=True)
os.makedirs(DOCSTORE_PATH, exist_ok=True)

client = QdrantClient(path=QDRANT_DB_PATH)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))
        },
        on_disk_payload=True, 
        hnsw_config=models.HnswConfigDiff(on_disk=True), 
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2)
    )

vector_store = QdrantVectorStore(
    client=client, 
    collection_name=COLLECTION_NAME, 
    embedding=embedding_model,
    sparse_embedding=sparse_embeddings,
    vector_name="dense",
    sparse_vector_name="sparse"
)

store = LocalFileStore(DOCSTORE_PATH)

# -------------------------------------------------------------
# 3. Setup the Splitters
# -------------------------------------------------------------
headers_to_split_on = getattr(config, "HEADERS_TO_SPLIT_ON", [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
    ("####", "Subsubsection"),
    ("#####", "Subsubsubection"),
    ("######", "Subsubsubsubection")
])

# 1. Markdown Splitter
parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

# 2. Size-Capped Parent Splitter (Prevents token overflow on massive sections)
parent_size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=getattr(config, "PARENT_CHUNK_SIZE", 1500),
    chunk_overlap=getattr(config, "PARENT_CHUNK_OVERLAP", 200),
    length_function=token_length,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 3. Granular Child Splitter (Restored exactly to your original parameters)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=getattr(config, "CHILD_CHUNK_SIZE", 400),          
    chunk_overlap=getattr(config, "CHILD_CHUNK_OVERLAP", 100),
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
    search_type="similarity", # Qdrant automatically fuses dense/sparse here            
    search_kwargs={
        "k": getattr(config, "RETRIEVAL_TOP_K", 20)  # Fixed: fetch_k removed for native similarity search
    }
)

banned_headers = getattr(config, "BANNED_HEADERS", [
    "references", 
    "bibliography", 
    "contributing authors", 
    "lead authors",
    "acknowledgements",
    "table of contents", 
    "list of figures",
    "list of tables",
    "data availability",
    "review editors",
    "chapter scientists",
    "coordinating lead authors",
    "this chapter should be cited as"
])

# -------------------------------------------------------------
# 4. Restored Helper Functions
# -------------------------------------------------------------
def build_safe_context(docs, max_tokens=3000):
    """
    Iterates through retrieved chunks, formats them as dictionaries,
    calculates exact Llama tokens, and enforces the token budget.
    """
    context_list = []
    total_tokens = 0
    
    print("\n[INFO] Calculating Context Tokens...")
    
    for i, doc in enumerate(docs, 1):
        doc_dict = {
            "document_index": i,
            "metadata": doc.metadata,
            "content": doc.page_content
        }
        
        doc_json_str = json.dumps(doc_dict, indent=2)
        doc_tokens = len(llm_tokenizer.encode(doc_json_str, add_special_tokens=False))
        
        if total_tokens + doc_tokens > max_tokens:
            print(f"       -> [SKIPPED] Document {i} ({doc_tokens} tokens) - Exceeds budget.")
            break 
            
        print(f"       -> [ADDED] Document {i}: {doc_tokens} tokens")
        context_list.append(doc_dict)
        total_tokens += doc_tokens
        
    print(f"[INFO] Final Context Size: {total_tokens}/{max_tokens} tokens.")
    return json.dumps(context_list, indent=2)


def enrich_visual_metadata(sections):
    """
    Scans chunk content for references to Figures and Tables.
    Restored to accept the section object directly.
    """
    fig_pattern = re.compile(r"(?:image|figure|fig\.?)\s*(\d+(?:[\w\.\-]*\w)?)", re.IGNORECASE)
    tbl_pattern = re.compile(r"(?:Table|Tbl\.?)\s*(\d+(?:[\w\.\-]*\w)?)", re.IGNORECASE)

    figures = list(set(fig_pattern.findall(sections.page_content)))
    tables = list(set(tbl_pattern.findall(sections.page_content)))

    return figures, tables

# -------------------------------------------------------------
# 5. Core Action Wrappers
# -------------------------------------------------------------
def process_and_ingest():
    search_path = "data/extracted_data/**/md/"
    folders = glob.glob(search_path)
    if not folders and hasattr(config, "EXTRACTED_DATA_DIR"):
        folders = glob.glob(os.path.join(str(config.EXTRACTED_DATA_DIR), "**/md/"))
        
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

            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=filepath)
                            )
                        ]
                    )
                )
            )
            
            # Step 1: Split into AST sections
            ast_sections = parent_splitter.split_text(doc.page_content)

            clean_ast_sections = []
            for section in ast_sections:
                # Flatten all header metadata into a single lowercase string
                header_text = " ".join(str(v) for v in section.metadata.values()).lower()
                
                # Tag the section based on whether it contains banned keywords
                if any(banned in header_text for banned in banned_headers):
                    section.metadata["content_tier"] = "noise"
                else:
                    section.metadata["content_tier"] = "core_science"
                    
                clean_ast_sections.append(section)

            # Now pass the tagged sections to your size-capper
            file_parent_docs = []
            for section in clean_ast_sections:
                bounded_parents = parent_size_splitter.split_documents([section])
                file_parent_docs.extend(bounded_parents)
            
            for idx, section in enumerate(file_parent_docs):
                figures, tables = enrich_visual_metadata(sections=section)
                parent_content_hash = hashlib.sha256(section.page_content.encode("utf-8")).hexdigest()
                
                chunk_id = f"{current_hash[:16]}_p{idx:04d}"

                section.metadata["doc_type"] = doc_type
                section.metadata["source"] = filepath
                section.metadata["has_figures"] = figures
                section.metadata["has_tables"] = tables
                section.metadata["version_hash"] = current_hash
                section.metadata["last_updated"] = formatted_date
                
                # Lineage
                section.metadata["parent_id"] = chunk_id
                section.metadata["parent_hash"] = parent_content_hash
                section.metadata["chunk_id"] = chunk_id
                
                if idx < len(file_parent_docs) - 1:
                    section.metadata["next_id"] = f"{current_hash[:16]}_p{idx + 1:04d}"

                all_parent_docs.append(section)

    if all_parent_docs:
        BATCH_SIZE = getattr(config, "INGESTION_BATCH_SIZE", 100)
        logging.info(f"Extracted {len(all_parent_docs)} Parent Sections. Starting Batched Ingestion...")
        
        for i in range(0, len(all_parent_docs), BATCH_SIZE):
            batch = all_parent_docs[i : i + BATCH_SIZE]
            batch_ids = [doc.metadata["chunk_id"] for doc in batch]
            retriever.add_documents(batch, ids=batch_ids)
            logging.info(f" -> Ingested batch {i//BATCH_SIZE + 1}/{(len(all_parent_docs)//BATCH_SIZE) + 1}")

        logging.info("Ingestion complete!")
        logging.info(f" -> Child Vectors searchable in Qdrant: {client.count(COLLECTION_NAME).count}")
    else:
        logging.warning("No documents found to ingest!")

if __name__ == "__main__":
    process_and_ingest()
    vector_store.client.close()
