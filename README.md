<<<<<<< HEAD
# VLM_RAG_Pipeline
This pipeline is to create the vlm rag pipeline.
=======
IPCC Scientific RAG Pipeline An industrial-grade Retrieval-Augmented Generation (RAG) pipeline engineered specifically for highly structured, dense scientific literature—specifically the IPCC climate reports.Unlike standard RAG tutorials that use naive semantic splitting, this architecture guarantees mathematical retrieval precision, hierarchical traceability, and LLM context safety by combining LangChain's ParentDocumentRetriever with dynamic forward-expansion and strict token budgeting.

Architecture & Key Innovation

1. 5-Tier AST (Abstract Syntax Tree) SplittingStandard chunking destroys scientific context. This pipeline parses documents based on their native Markdown headers (from # Chapter down to ##### Subsubsubection). The LLM receives text grounded in its exact location in the book, allowing for highly accurate citations.

2. Parent-Child Retrieval StrategyChild Chunks (200 tokens): Highly granular vectors optimized for dense mathematical and semantic search.Parent Documents (1,000+ tokens): The full narrative section containing the child chunk is passed to the LLM to provide surrounding scientific context.

3. Dynamic Forward Context ExpansionScientific concepts often span multiple paragraphs. If the retrieved Parent Document is below a safe token threshold, the pipeline automatically traverses the local docstore and chains subsequent sections forward until the LLM has enough context to answer accurately—without fetching irrelevant preceding summaries.

4. Asymmetric Scientific EmbeddingsPowered by BAAI/bge-large-en-v1.

5. The pipeline applies asymmetric instruction-tuning, embedding the IPCC documents densely while prepending "Represent this sentence for searching relevant passages: " to user queries at inference time.5. Strict Token BudgetingModern LLMs suffer from the "Lost in the Middle" effect when fed too much context. This pipeline dynamically measures exact tokens using the hf-internal-testing/llama-tokenizer and enforces a strict budget (e.g., 3,000 tokens) to guarantee zero hallucinations and fast inference times. 

Tech Stack

ComponentTechnologyVector DatabaseQdrant (Local, Disk-Optimized via on_disk_payload)Embedding ModelHuggingFace (BAAI/bge-large-en-v1.5)LLM EngineOllama (Llama-2 / Llama-3)OrchestrationLangChain / LangChain-QdrantStorageLocalFileStore (Parent Docs) 

Project StructurePlaintext├── data/
│   └── extracted_data/
│       └── chapter_3/
│           └── md/
│               └── *_hydrated.md      # Markdown files with VLM-extracted figures injected
├── vector_database/
│   ├── quadrant_database/
│   │   ├── qdrant_db/                 # Qdrant disk storage
│   │   └── docstore/                  # LocalFileStore bytes (Parent Docs)
├── chunking_vectordb.py               # Ingestion, AST splitting, and Vectorization logic
├── inference.py                       # Retrieval, Dynamic Expansion, Router, and LLM generation
├── dynamic_splitter.py                # Custom expansion and token math logic
├── file_versioning.py                 # File hashing for incremental updates
└── README.md


Installation & Setup

1. PrerequisitesPython 3.10+Ollama installed and running locally.

2. Install DependenciesBashpip install -r requirements.txt
(Ensure you have langchain, langchain-qdrant, langchain-huggingface, qdrant-client, and transformers installed).

3. Pull the Local LLMBashollama pull llama2

Usage 

Phase 1: IngestionRun the pipeline script with the ingestion function enabled to process your hydrated Markdown files, generate the 1024-dimensional vectors, and store the parent documents.The ingestion script is idempotent. It hashes files and safely deletes old vectors before replacing them, meaning you can run it incrementally without corrupting the database.Bashpython chunking_vectordb.py

Phase 2: InferenceRun the inference script to test the pipeline. The script features a lightweight semantic router that automatically bypasses vector search for global queries (e.g., "List all figures") and routes standard scientific questions through the Qdrant + Dynamic Expansion flow.Bashpython inference.py

Example Output:Plaintext[USER]: What is the Human Influence on the Cryosphere?

[INFO] Retrieving top parent documents from Qdrant...
       -> [EXPANDING FORWARD] Chunk 'a1b2c3d4_0014' (350 tokens)...
          [+] Appended 'a1b2c3d4_0015'. Total size: 850 tokens.
          [+] Appended 'a1b2c3d4_0016'. Total size: 1200 tokens.

[INFO] Calculating Context Tokens...
       -> Document 1: 1200 tokens
       -> [ADDED] Document 1: 1200 tokens
[INFO] Final Context Size: 1200/3000 tokens.

[INFO] Generating response with Llama-2...

--- FINAL ANSWER ---
Based on Document 1 (Section 3.4 - Cryosphere), human influence has very likely contributed to the melting of glaciers and decreases in Arctic sea ice since the late 20th century...
>>>>>>> dev
