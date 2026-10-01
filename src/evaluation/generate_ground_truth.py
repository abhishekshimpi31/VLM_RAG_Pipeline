import os
import glob
import json
import logging
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.load import loads

from ..ingestion.pdf_image_rendering import clean_markdown_text
from ..indexing.chunking_vectordb import store, banned_headers
import config


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. Configuration
BASE_DATA_DIR = "data/extracted_data/**/md/"
OUTPUT_FILE = getattr(config, "GROUND_TRUTH_DATASET_PATH", "data/extracted_data/testing_data/ground_truth_dataset.json")
NUM_CHUNKS_TO_PROCESS = 500 # Limit this during testing so it doesn't run for hours

# Use a stronger local model if possible, and enforce JSON output
print("[INFO] Loading Ollama for Synthetic Generation...")
llm = ChatOllama(model=getattr(config, "LLM_MODEL_NAME", "llama3"), temperature=0.1, format="json")

# 2. Setup Splitter (Must match your ingestion pipeline)
headers_to_split_on = getattr(config, "HEADERS_TO_SPLIT_ON", [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
    ("####", "Subsubsection"),
    ("#####", "Subsubsubection")
])
parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

# 3. Define the Teacher Prompt
# We ask for a JSON array of Question/Answer pairs.
prompt = PromptTemplate(
    template="""You are an expert climate scientist creating a reading comprehension test based on the IPCC report.
    Given the following document section, generate 2 highly specific questions and their exact, factual answers.
    The questions must require understanding of the scientific data or concepts in the text.
    
    Return ONLY a valid JSON object with a single key called "qa_pairs" which contains an array of the questions and answers.
    Example format:
    {{
      "qa_pairs": [
        {{"question": "What is the primary driver of...", "ground_truth": "The primary driver is..."}},
        {{"question": "According to the section on...", "ground_truth": "The data shows a 2% increase..."}}
      ]
    }}

    DOCUMENT TEXT:
    {context}
    """,
    input_variables=["context"]
)

# Create the generation chain
generation_chain = prompt | llm | JsonOutputParser()

def generate_dataset():
    # 4. Load Documents
    folders = glob.glob(BASE_DATA_DIR)
    if not folders and hasattr(config, "EXTRACTED_DATA_DIR"):
        folders = glob.glob(os.path.join(str(config.EXTRACTED_DATA_DIR), "**/md/"))
        
    all_sections = []
    
    print("[INFO] Loading and splitting markdown files...")
    for folder in folders:
        loader = DirectoryLoader(folder, glob="*_hydrated.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()
        for doc in docs:
            sections = parent_splitter.split_text(doc.page_content)
            all_sections.extend(sections)
            
    # Filter out tiny sections (e.g., just a title) that can't support good questions
    valid_sections = [sec for sec in all_sections if len(sec.page_content.split()) > 100]
    valid_sections = valid_sections[:NUM_CHUNKS_TO_PROCESS]
    
    print(f"[INFO] Generating Q&A pairs for {len(valid_sections)} chunks...\n")
    
    dataset = []
    
    # 5. Generate Q&A Pairs
    for i, section in enumerate(tqdm(valid_sections, desc="Generating Ground Truth")):
        try:
            # ---> APPLY THE CLEANER HERE <---
            clean_text = clean_markdown_text(section.page_content)
            
            # Skip if the chunk was just a bunch of empty tables and is now too short
            if len(clean_text.split()) < 30:
                continue

            # 1. Generate the raw parsed JSON using the CLEANED text
            result = generation_chain.invoke({"context": clean_text})
            
            # 2. Safely extract the list of pairs (handles both dicts and naked lists)
            if isinstance(result, dict) and "qa_pairs" in result:
                qa_pairs = result["qa_pairs"]
            elif isinstance(result, dict):
                qa_pairs = next((v for v in result.values() if isinstance(v, list)), [result])
            elif isinstance(result, list):
                qa_pairs = result
            else:
                logging.warning(f"Unexpected output type for chunk {i}: {type(result)}")
                continue
            
            # 3. Map the results into the Ragas-compatible schema
            for pair in qa_pairs:
                if not isinstance(pair, dict): 
                    continue 
                
                dataset.append({
                    "question": pair.get("question"),
                    "ground_truth": pair.get("ground_truth"),
                    # Save the CLEANED text to the final JSON
                    "reference_context": clean_text, 
                    "metadata": section.metadata
                })

            print(dataset[-1])  # Debug: Show the last generated pair for verification
                
        except Exception as e:
            logging.warning(f"Failed to process chunk {i}: {e}")
            continue

    # 6. Save to Disk
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n[SUCCESS] Generated {len(dataset)} evaluation pairs. Saved to {OUTPUT_FILE}")


# 1. Configuration for Retrieval Benchmark
RETRIEVAL_OUTPUT_FILE = getattr(config, "RETRIEVAL_BENCHMARK_PATH", "data/extracted_data/testing_data/retrieval_benchmark.json")

# 2. Simplified Teacher Prompt
retrieval_prompt = PromptTemplate(
    template="""You are an expert climate scientist creating a search benchmark for an IPCC report.
    Read the following document section and generate 2 highly specific questions that can ONLY be answered by retrieving this exact text.
    
    Return ONLY a valid JSON object with a single key called "questions" containing an array of strings.
    Example format:
    {{
      "questions": [
        "What is the primary driver of...",
        "How does the data in this section show a 2% increase in..."
      ]
    }}

    DOCUMENT TEXT:
    {context}
    """,
    input_variables=["context"]
)

retrieval_generation_chain = retrieval_prompt | llm | JsonOutputParser()

def is_core_science(doc) -> bool:
    """Evaluates if a document is core science using metadata tag or the shared banned_headers list."""
    tier = doc.metadata.get("content_tier")
    if tier:
        return tier == "core_science"

    # Fallback to the single source of truth for banned headers
    header_context = " ".join(str(v) for v in doc.metadata.values()).lower()
    return not any(term in header_context for term in banned_headers)


def generate_retriver_dataset():
    print("[INFO] Fetching parent documents directly from the docstore...")

    keys = list(store.yield_keys())
    if not keys:
        print("[ERROR] Docstore is empty! Please run your ingestion script first.")
        return

    raw_parent_docs = store.mget(keys)
    parent_docs = []

    for raw_doc in raw_parent_docs:
        if raw_doc:
            try:
                parent_docs.append(loads(raw_doc.decode("utf-8")))
            except Exception as e:
                logging.warning(f"Failed to deserialize a document: {e}")

    # Filter candidates strictly before applying the slice limit
    valid_parents = [
        doc
        for doc in parent_docs
        if doc and len(doc.page_content.split()) > 100 and is_core_science(doc)
    ][:NUM_CHUNKS_TO_PROCESS]

    print(
        f"[INFO] Generating queries for {len(valid_parents)} valid core_science parent documents...\n"
    )

    dataset = []

    for i, section in enumerate(tqdm(valid_parents, desc="Generating Benchmark")):
        try:
            clean_text = clean_markdown_text(section.page_content)
            if len(clean_text.split()) < 30:
                continue

            result = retrieval_generation_chain.invoke({"context": clean_text})
            questions = result.get("questions", [])
            if not isinstance(questions, list):
                continue

            target_id = section.metadata.get("parent_id") or section.metadata.get(
                "chunk_id"
            )

            for q in questions:
                dataset.append(
                    {"question": q, "target_parent_id": target_id}
                )

        except Exception as e:
            logging.warning(f"Failed to process docstore chunk {i}: {e}")
            continue

    os.makedirs(os.path.dirname(RETRIEVAL_OUTPUT_FILE), exist_ok=True)
    with open(RETRIEVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(
        f"\n[SUCCESS] Generated {len(dataset)} evaluation queries. Saved to {RETRIEVAL_OUTPUT_FILE}"
    )

if __name__ == "__main__":
    # generate_dataset()
    generate_retriver_dataset()
