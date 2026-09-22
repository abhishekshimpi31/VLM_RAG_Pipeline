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

from pdf_image_rendering import clean_markdown_text

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. Configuration
BASE_DATA_DIR = "data/extracted_data/**/md/"
OUTPUT_FILE = "data/extracted_data/testing_data/ground_truth_dataset.json"
NUM_CHUNKS_TO_PROCESS = 500 # Limit this during testing so it doesn't run for hours

# Use a stronger local model if possible, and enforce JSON output
print("[INFO] Loading Ollama for Synthetic Generation...")
llm = ChatOllama(model="llama3", temperature=0.1, format="json")

# 2. Setup Splitter (Must match your ingestion pipeline)
headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
    ("####", "Subsubsection"),
    ("#####", "Subsubsubection")
]
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
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n[SUCCESS] Generated {len(dataset)} evaluation pairs. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()