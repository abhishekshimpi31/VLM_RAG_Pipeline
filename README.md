# IPCC Scientific RAG Pipeline

An industrial-grade Retrieval-Augmented Generation (RAG) pipeline engineered specifically for highly structured, dense scientific literature—specifically the **IPCC Climate Assessment Reports**.

Unlike standard RAG pipelines that use naive semantic or fixed-character splitting, this architecture guarantees mathematical retrieval precision, hierarchical traceability, and LLM context safety by combining LangChain's `ParentDocumentRetriever` with dynamic forward-expansion, multi-modal figure hydration, and strict token budgeting.

---

## System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          IPCC PDF Assessment Reports         │
                    └──────────────────────┬───────────────────────┘
                                           │
                        [PyMuPDF & PyMuPDF4LLM Extraction]
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
   [Figure / Image Detection]                                   [Section AST Parser]
             │                                                           │
   [Qwen2-VL-7B Ingestion]                                               │
   (Extract axes, units, trends)                                         │
             │                                                           │
             ▼                                                           │
   [JSON Chapter Registry]                                               │
             │                                                           │
             └─────────────────────────────┬─────────────────────────────┘
                                           ▼
                            [Markdown Hydration Stage]
                      (Injects structured visual blockquotes)
                                           │
                                           ▼
                           [5-Tier Markdown AST Splitter]
                              (# Chapter down to ######)
                                           │
                                           ▼
                      [Parent Size Capper (1500 tokens)]
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
     [Child Splitter (400 tok)]                         [Parent Docs LocalFileStore]
                 │                                                   │
                 ▼                                                   │
     [Qdrant Hybrid Vector Store]                                    │
     (Dense BGE + Sparse BM25)                                       │
                 │                                                   │
                 │ ◄────── User Query (Asymmetric BGE) ──────────────┤
                 ▼                                                   │
     [Top-K Child Candidates]                                        │
                 │                                                   │
                 ▼                                                   │
     [BGE Cross-Encoder Reranker]                                     │
                 │                                                   │
                 ▼                                                   ▼
     [Deduplicated Parent Lookup] ──────────────────────► [Fetch from Docstore]
                                                                     │
                                                                     ▼
                                                         [Dynamic Forward Expansion]
                                                         (Chains neighbor sections)
                                                                     │
                                                                     ▼
                                                          [LLaMA Token Budgeting]
                                                          (Strict <=5000 tokens)
                                                                     │
                                                                     ▼
                                                          [Grounded Prompt Synthesis]
                                                          (Zero outside knowledge)
                                                                     │
                                                                     ▼
                                                          [Ollama LLaMA-3.1 Answer]
```

---

## Project Directory Structure

```plaintext
Rag_Pipeline/
├── config.py                          # Centralized dynamic configuration & prompts
├── pyproject.toml                     # Python packaging specification
├── requirements.txt                   # Complete production dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Version control exclusions
│
├── src/                               # Modular Source Package
│   ├── common/                        # Shared file utilities & hashing
│   │   ├── __init__.py
|   |   ├── model_dependency.py        # Centralised model initialisation 
│   │   └── file_versioning.py         # File hashing, TTL retention, safe chapter IDs
│   ├── ingestion/                     # Multi-modal extraction & Markdown hydration
│   │   ├── __init__.py
│   │   ├── pdf_image_rendering.py     # Snapshot rendering, dHash fingerprinting, caption regex
│   │   ├── context_generator.py       # Visual queueing, caching & orphan cleanup
│   │   ├── manifest_mdfile_generation.py # AST document parsing & manifest generation
│   │   ├── vlm_model.py               # Qwen2-VL-7B 8-bit multi-modal extraction
│   │   └── image_context_ingestion.py # In-place Markdown blockquote hydration
│   ├── indexing/                      # Chunking, embeddings & vector store
│   │   ├── __init__.py
│   │   ├── dynamic_splitter.py        # Tokenizer math & dynamic forward neighbor expansion
│   │   └── chunking_vectordb.py       # AST splitters, Qdrant hybrid setup, batched ingestion
│   ├── retrieval/                     # Search, reranking & LLM generation
│   │   ├── __init__.py
│   │   ├── guardrails.py              # Pythoon based guardrails to protect the backend
│   │   ├── semantic_cache.py          # Cached alredy executed queries to reduced the latency and compute
│   │   └── inference.py               # Child reranker, safe token budgeting, grounded QA
│   └── evaluation/                    # Benchmarking & automated metrics
│       ├── __init__.py
│       ├── generate_ground_truth.py   # Synthetic QA & retrieval dataset generator
│       ├── evaluate_retrieval.py      # Deterministic IR metrics (MRR, NDCG, Precision, HitRate)
│       └── evaluate_pipeline.py       # End-to-end evaluation with Ragas & LLM-as-a-judge
│
├── scripts/                           # Production CLI Executables
│   ├── run_ingestion.py               # Batch-embeds and indexes documents into Qdrant
│   ├── run_inference.py               # Command-line query inference runner
│   ├── run_vlm.py                     # Runs Qwen2-VL on pending image manifests
│   ├── run_hydration.py               # Hydrates raw Markdown with VLM figure analyses
│   ├── run_retrieval_benchmark.py     # Executes deterministic IR benchmark evaluation
│   └── run_pipeline_eval.py           # Runs complete Ragas evaluation suite
│
├── tests/                             # Unit & Sanity Test Suite
│   ├── __init__.py
│   └── test_sanity.py
│
├── data/                              # Data directories (PDFs, extracted markdown, testing datasets)
└── vector_database/                   # Local persistent Qdrant & docstore storage
```

---

## Key Features

1. **5-Tier AST (Abstract Syntax Tree) Splitting**:
   Parses documents according to native Markdown headers (from `# Chapter` down to `###### Subsubsubsubection`), preserving exact hierarchical document context and citation breadcrumbs.
2. **Parent-Child Retrieval Strategy**:
   * **Child Chunks (~400 tokens)**: Granular vectors optimized for dense mathematical and sparse keyword search.
   * **Parent Documents (~1,500 tokens)**: Full narrative sections containing child chunks are fetched from `LocalFileStore` to provide comprehensive context to the LLM.
3. **Multi-Modal Visual Data Hydration**:
   Renders PDF page graphics, hashes them via perceptual `dHash`, and uses `Qwen2-VL-7B` to extract axes, units, legends, and quantitative trends. The extracted analysis is injected directly into the document Markdown as analytical blockquotes.
4. **Hybrid Dense + Sparse Vector Search**:
   Combines dense semantic representations (`BAAI/bge-large-en-v1.5`) with sparse BM25 (`FastEmbedSparse`) in Qdrant, ensuring that both high-level concepts and exact scientific acronyms/IDs are retrievable.
5. **Dynamic Forward Context Expansion**:
   If a retrieved parent document falls below a minimum token threshold, the pipeline automatically traverses the local docstore along `next_id` linked-list chains, appending forward neighbor sections without grabbing irrelevant preceding text.
6. **BGE Cross-Encoder Reranking**:
   Employs `BAAI/bge-reranker-large` over retrieved child chunks, deduplicating to the most relevant parent documents and avoiding cross-encoder token truncation.
7. **Strict Token Budgeting & Grounded Synthesis**:
   Enforces a strict token ceiling using the LLaMA tokenizer to avoid LLM "lost-in-the-middle" degradation, paired with a prompt requiring 100% factual grounding and bracketed `[index]` citations.

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.ai/) installed and running locally
* CUDA-capable GPU (recommended for local VLM / embedding acceleration)

### 2. Pull Local LLM Model
```bash
ollama pull llama3.1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install in editable package mode:
```bash
pip install -e .
```

---

## Usage & Execution

### Phase 1: Ingestion & Vectorization
Process hydrated Markdown files, generate dense & sparse vectors, and index parent documents:
```bash
python scripts/run_ingestion.py
```

### Phase 2: QA Inference & Interactive Chatbot
#### 1. CLI Inference
Query the scientific literature directly from the terminal:
```bash
python scripts/run_inference.py --query "What is the likely range of human-induced warming?"
```

### Phase 3: Multi-Modal Figure Extraction (Optional / Advanced)
1. **Extract figures and create manifests**:
   ```bash
   python scripts/run_ingestion.py
   ```
2. **Run VLM inference**:
   ```bash
   python scripts/run_vlm.py --batch-size 8
   ```
3. **Hydrate Markdown with figure data**:
   ```bash
   python scripts/run_hydration.py
   ```

### Phase 4: Evaluation & Benchmarking
* **Evaluate Retrieval (Deterministic IR Metrics - MRR, NDCG, HitRate)**:
  ```bash
  python scripts/run_retrieval_benchmark.py
  ```
* **Evaluate End-to-End Pipeline (RAGAS - Faithfulness, Relevancy)**:
  ```bash
  python scripts/run_pipeline_eval.py
  ```

---

## Running Tests

Run the test suite to verify pipeline utilities, config loading, FastAPI endpoints, and metric computations:
```bash
python -m unittest tests/test_sanity.py
```