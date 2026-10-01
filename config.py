import os
from pathlib import Path

# Load environment variables from .env file if dotenv is installed

from dotenv import load_dotenv

load_dotenv(override=True)


# ==============================================================================
# 1. DIRECTORY PATHS (Dynamically resolved relative to project root)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
PDF_DIR = Path(os.getenv("PDF_DIR", DATA_DIR / "pdf_files"))
EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_DIR", DATA_DIR / "extracted_data"))
TESTING_DATA_DIR = Path(os.getenv("TESTING_DATA_DIR", EXTRACTED_DATA_DIR / "testing_data"))

VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", BASE_DIR / "vector_database"))
QDRANT_DB_PATH = str(Path(os.getenv("QDRANT_DB_PATH", VECTOR_DB_DIR / "quadrant_database" / "qdrant_db")))
DOCSTORE_PATH = str(Path(os.getenv("DOCSTORE_PATH", VECTOR_DB_DIR / "quadrant_database" / "docstore")))
CHROMA_DB_PATH = str(Path(os.getenv("CHROMA_DB_PATH", VECTOR_DB_DIR / "chromadb" / "vector_db")))

# ==============================================================================
# 2. MODEL CONFIGURATIONS
# ==============================================================================
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ipcc_hybrid_chunks")

# Embedding Models
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")
MINILM_MODEL_NAME = os.getenv("MINILM_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Tokenizers
EMBEDDING_TOKENIZER_NAME = os.getenv("EMBEDDING_TOKENIZER_NAME", "BAAI/bge-large-en-v1.5")
LLM_TOKENIZER_NAME = os.getenv("LLM_TOKENIZER_NAME", "hf-internal-testing/llama-tokenizer")

# Cross-Encoder Reranker
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")

# Vision Language Model (VLM)
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

# Local LLM (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8000"))

# ==============================================================================
# 3. SPLITTER & RETRIEVAL HYPERPARAMETERS (Preserved Exactly)
# ==============================================================================
HEADERS_TO_SPLIT_ON = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
    ("####", "Subsubsection"),
    ("#####", "Subsubsubection"),
    ("######", "Subsubsubsubection")
]

PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200

CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 100

EXPANSION_MIN_TOKENS = 1000
EXPANSION_CEILING_TOKENS = 1200
MAX_SAFE_CONTEXT_TOKENS = 5000

INGESTION_BATCH_SIZE = 100
RETRIEVAL_TOP_K = 20
RERANKER_TOP_K = 3

BANNED_HEADERS = [
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
]

# ==============================================================================
# 4. BENCHMARK & EVALUATION PATHS
# ==============================================================================
GROUND_TRUTH_DATASET_PATH = str(TESTING_DATA_DIR / "ground_truth_dataset.json")
RETRIEVAL_BENCHMARK_PATH = str(TESTING_DATA_DIR / "retrieval_benchmark.json")
RETRIEVAL_METRICS_OUTPUT_CSV = str(TESTING_DATA_DIR / "retrieval_metrics.csv")
EVALUATION_RESULTS_OUTPUT_CSV = str(TESTING_DATA_DIR / "evaluation_results.csv")

# ==============================================================================
# 5. VLM PROMPTS (Preserved Exactly)
# ==============================================================================
SYSTEM_PROMPT = """You are an expert scientific data extractor and data analysis.
You will receive an image of a document page and the official captions corresponding to the figures on that page.

YOUR OBJECTIVE:
Visually scan the image for any charts, graphs, data tables, or scientific diagrams. Extract and describe them in deep analytical detail.

CORE RULES:
1. DEEP ANALYTICAL EXTRACTION: Detail axes, units, legends, sub-panels (e.g., (a), (b), (c)), baselines, trends, and specific numerical/statistical findings.
2. CAPTION ALIGNMENT: Directly correlate visual panels to their respective figure numbers and caption descriptions.
3. MULTIPLE FIGURES: If there are multiple charts or figures on the page, describe EACH one separately and comprehensively. Do not group them together.
5. ANALYTICAL DEPTH: For each chart, identify the chart type, axes, units, legends, and key data trends. Treat the figure as a structured repository of scientific data.

FORMATTING:
Describe each figure clearly, separating multiple figures with line breaks. Maintain a professional, highly precise scientific tone."""


user_prompt = """### SURROUNDING DOCUMENT CONTEXT
-----------------------------------------
{captions_text}
-----------------------------------------

### YOUR EXTRACTION TASK
Analyze the provided page image alongside the surrounding document context above. Perform a rigorous extraction of the scientific figures, charts, or complex data graphics shown in the image.

### EXTRACTION CHECKLIST
For each distinct chart or figure found in the image, extract and describe:
1. Identification: The Figure/Table number and its exact title/caption (cross-reference the surrounding text to find this).
2. Visual Structure: The chart type (e.g., scatter plot, bar chart, map), the variables on the X and Y axes, and the units of measurement.
3. Legends & Categories: Explain the legend, color-coding, or scenarios (e.g., SSP1-2.6, SSP5-8.5).
4. Quantitative Data & Trends: Describe the key numerical ranges, trajectories, inflection points, and confidence intervals.
5. Multi-panel breakdown: If the graphic contains sub-panels (e.g., (a), (b), (c)), analyze each panel distinctly.

### STRICT OUTPUT CONSTRAINTS
- You MUST start your exact response strictly with: 'ID: {image_id} - '
- If the image contains no charts, graphs, or data figures, you MUST output exactly and only: "NO_CHARTS_FOUND"
"""

# ==============================================================================
# 6. INFERENCE & RAG QA PROMPTS
# ==============================================================================
QUERY_EXPANSION_PROMPT_TEMPLATE = """You are a strict search optimization assistant for a professional retrieval-augmented generation (RAG) system.
Your task is to evaluate and potentially rewrite the user's search query to maximize retrieval accuracy from a vector database.

RULES:
1. Conditional Modification: ONLY modify the query if it is overly sparse, lacks context, or uses colloquial terminology. If the query is already detailed, well-formulated, and uses proper terminology, return it EXACTLY as-is without any changes.
2. Vocabulary Normalization: Translate colloquial visual or structural terms (e.g., "image", "picture", "graph", "chart") into standard formal nomenclature (e.g., "Figure", "Table", "Section") based on typical professional formatting.
3. Exact Identifier Preservation: Keep all numbers, alphanumeric IDs, dates, and proper nouns (e.g., "3.20", "3.SM.1", "Q3", "John Doe") EXACTLY intact. 
4. Natural Semantic Expansion: If rewriting a sparse query (e.g., "What does Figure 3.20 show?"), do not just append disconnected keywords. Instead, rewrite it into a natural, context-rich sentence (e.g., "Describe the data, findings, and analytical context presented in Figure 3.20.").
5. Strict Context Adherence: You must only expand on the exact topic requested. You are strictly forbidden from attempting to answer the user's question, guessing the topic, or introducing external facts. 
6. Output Format: Respond ONLY in valid JSON format with a single key "expanded_query".

USER QUERY:
{query}

JSON RESPONSE:"""

QA_PROMPT_TEMPLATE = """You are an expert analytical assistant and strict fact-synthesizer. Your sole purpose is to extract and summarize information EXCLUSIVELY from the provided source documents.
You will be provided with a JSON array of source documents. Each document contains 'metadata', 'content', and a 'document_index'.

CRITICAL INSTRUCTIONS FOR MAXIMUM FAITHFULNESS:
1. ZERO OUTSIDE KNOWLEDGE: Your answer must be 100% grounded in the provided JSON context. Do not include any external knowledge, assumptions, logical leaps, or explanations that are not explicitly stated in the text. Even if you know a fact to be true, if it is not in the context, DO NOT mention it.
2. EXACT QUANTIFICATION: Quote all numerical values, percentages, dates, and calibrated uncertainty terms (e.g., "high confidence", "very likely") exactly as they appear in the source. Do not round numbers or approximate.
3. HANDLING MISSING INFORMATION: 
   - If the context completely lacks the facts needed to answer the question, you must state EXACTLY: "I do not have enough information in the provided context to answer this question."
   - If the context only partially answers the question, provide ONLY the information present in the text and do not guess the rest.
4. MANDATORY CITATIONS: Every single sentence or distinct factual claim you write MUST be immediately followed by its source citation using the 'document_index' in square brackets (e.g., [1]). 
   - STRICT RULE: If a sentence cannot be directly cited to the provided text, you are not allowed to write that sentence.
   - Example: "Global mean sea level increased by 0.20m between 1901 and 2018 [1]. This rate is faster than any preceding century in at least 3000 years [2]."

JSON CONTEXT:
{context}

USER QUESTION: 
{question}

Synthesize a direct, highly accurate answer based ONLY on the context above. Include citations for every claim:"""

CONVERSATIONAL_QA_PROMPT_TEMPLATE = """You are an expert analytical assistant and strict fact-synthesizer. Your sole purpose is to extract and summarize information EXCLUSIVELY from the provided source documents.
You will be provided with a JSON array of source documents. Each document contains 'metadata', 'content', and a 'document_index'.

CRITICAL INSTRUCTIONS FOR MAXIMUM FAITHFULNESS:
1. ZERO OUTSIDE KNOWLEDGE: Your answer must be 100% grounded in the provided JSON context. Do not include any external knowledge, assumptions, logical leaps, or explanations that are not explicitly stated in the text. Even if you know a fact to be true, if it is not in the context, DO NOT mention it.
2. EXACT QUANTIFICATION: Quote all numerical values, percentages, dates, and calibrated uncertainty terms (e.g., "high confidence", "very likely") exactly as they appear in the source. Do not round numbers or approximate.
3. HANDLING MISSING INFORMATION: 
   - If the context completely lacks the facts needed to answer the question, you must state EXACTLY: "I do not have enough information in the provided context to answer this question."
   - If the context only partially answers the question, provide ONLY the information present in the text and do not guess the rest.
4. MANDATORY CITATIONS: Every single sentence or distinct factual claim you write MUST be immediately followed by its source citation using the 'document_index' in square brackets (e.g., [1]).
5. CONVERSATION CONTEXT: Use the prior conversation history to understand follow-up references, pronouns, or clarifications, but ensure all newly asserted facts are cited from the JSON CONTEXT below.

JSON CONTEXT:
{context}

PRIOR CONVERSATION HISTORY:
{chat_history}

CURRENT USER QUESTION: 
{question}

Synthesize a direct, highly accurate answer based ONLY on the context above. Include citations for every claim:"""

CONDENSE_QUESTION_PROMPT_TEMPLATE = """Given the following conversation history and follow-up question, rephrase the follow-up question to be a standalone, context-rich scientific search query for document retrieval.
Keep all technical terms, figure numbers, chapter numbers, and variables intact. Do NOT answer the question; only return the reformulated query.

CHAT HISTORY:
{chat_history}

FOLLOW-UP QUESTION:
{question}

STANDALONE SEARCH QUERY:"""