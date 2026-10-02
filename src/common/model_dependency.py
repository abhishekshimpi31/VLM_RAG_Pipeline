"""
Centralized Model Management & Initialization Module
====================================================
Provides centralized, cached initialization for all AI models used across
the scientific RAG pipeline. Supports both direct lazy-loaded imports and 
explicit factory functions.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Fallback-aware config import
try:
    from src import config
except ImportError:
    config = None

# Registry caches for singleton instances
_MODELS: Dict[str, Any] = {}
_TOKENIZERS: Dict[str, Any] = {}


def _get_config(key: str, default: Any) -> Any:
    """Helper to safely retrieve attributes from the config module or env variables."""
    if config and hasattr(config, key):
        return getattr(config, key)
    return os.getenv(key, default)


def _load_tokenizer(model_name: str, force_reload: bool) -> Any:
    """Internal helper to load and cache HuggingFace tokenizers."""
    cache_key = f"tok_{model_name}"
    if not force_reload and cache_key in _TOKENIZERS:
        return _TOKENIZERS[cache_key]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    _TOKENIZERS[cache_key] = tok
    return tok


# ==============================================================================
# 1. EMBEDDING MODELS (DENSE & SPARSE)
# ==============================================================================
def get_embedding_model(
    model_name: Optional[str] = None,
    normalize_embeddings: bool = True,
    query_prompt: Optional[str] = "Represent this sentence for searching relevant passages: ",
    force_reload: bool = False
):
    """Initializes and caches the dense HuggingFace embedding model."""
    model_name = model_name or _get_config("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    cache_key = f"dense_embed_{model_name}_{normalize_embeddings}"

    if not force_reload and cache_key in _MODELS:
        return _MODELS[cache_key]

    logger.info(f"[MODELS] Loading Dense Embedding Model: {model_name}...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    embed_instance = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={'normalize_embeddings': normalize_embeddings},
        query_encode_kwargs={
            'normalize_embeddings': normalize_embeddings,
            **({'prompt': query_prompt} if query_prompt else {})
        }
    )
    _MODELS[cache_key] = embed_instance
    return embed_instance


def get_sparse_embedding_model(model_name: Optional[str] = None, force_reload: bool = False):
    """Initializes and caches the sparse BM25 FastEmbed model for hybrid search."""
    model_name = model_name or _get_config("SPARSE_MODEL_NAME", "Qdrant/bm25")
    cache_key = f"sparse_embed_{model_name}"

    if not force_reload and cache_key in _MODELS:
        return _MODELS[cache_key]

    logger.info(f"[MODELS] Loading Sparse Embedding Model: {model_name}...")
    from langchain_qdrant import FastEmbedSparse

    sparse_instance = FastEmbedSparse(model_name=model_name)
    _MODELS[cache_key] = sparse_instance
    return sparse_instance


# ==============================================================================
# 2. CROSS-ENCODER RERANKER
# ==============================================================================
def get_reranker(model_name: Optional[str] = None, use_fp16: Optional[bool] = None, force_reload: bool = False):
    """Initializes and caches the BAAI FlagReranker cross-encoder model."""
    model_name = model_name or _get_config("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
    use_fp16 = True if use_fp16 is None else use_fp16
    cache_key = f"reranker_{model_name}_{use_fp16}"

    if not force_reload and cache_key in _MODELS:
        return _MODELS[cache_key]

    logger.info(f"[MODELS] Loading FlagReranker ({model_name}, use_fp16={use_fp16})...")
    from FlagEmbedding import FlagReranker

    try:
        reranker_instance = FlagReranker(model_name, use_fp16=use_fp16)
    except Exception as e:
        if use_fp16:
            logger.warning(f"[MODELS] FlagReranker fp16 init failed ({e}), falling back to fp32...")
            reranker_instance = FlagReranker(model_name, use_fp16=False)
        else:
            raise

    _MODELS[cache_key] = reranker_instance
    return reranker_instance


# ==============================================================================
# 3. LARGE LANGUAGE MODELS (LLM & CHAT LLM)
# ==============================================================================
def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
    force_reload: bool = False
):
    """Initializes and caches the Ollama text generation LLM."""
    model_name = model_name or _get_config("LLM_MODEL_NAME", "llama3.1")
    temperature = temperature if temperature is not None else float(_get_config("LLM_TEMPERATURE", 0.1))
    num_ctx = num_ctx if num_ctx is not None else int(_get_config("LLM_NUM_CTX", 8000))
    base_url = base_url or _get_config("OLLAMA_BASE_URL", "http://localhost:11434")

    cache_key = f"llm_{model_name}_{temperature}_{num_ctx}_{base_url}"

    if not force_reload and cache_key in _MODELS:
        return _MODELS[cache_key]

    logger.info(f"[MODELS] Loading Ollama LLM: {model_name} (temp={temperature}, ctx={num_ctx})...")
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        from langchain_ollama import OllamaLLM as Ollama

    llm_instance = Ollama(
        model=model_name, temperature=temperature, num_ctx=num_ctx,
        base_url=base_url, verbose=verbose
    )
    _MODELS[cache_key] = llm_instance
    return llm_instance


def get_chat_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
    format: Optional[str] = None,
    base_url: Optional[str] = None,
    force_reload: bool = False
):
    """Initializes and caches the ChatOllama model."""
    model_name = model_name or _get_config("LLM_MODEL_NAME", "llama3.1")
    temperature = temperature if temperature is not None else 0.0
    num_ctx = num_ctx if num_ctx is not None else int(_get_config("LLM_NUM_CTX", 8000))
    base_url = base_url or _get_config("OLLAMA_BASE_URL", "http://localhost:11434")

    cache_key = f"chat_llm_{model_name}_{temperature}_{num_ctx}_{format}_{base_url}"

    if not force_reload and cache_key in _MODELS:
        return _MODELS[cache_key]

    logger.info(f"[MODELS] Loading ChatOllama: {model_name} (temp={temperature}, format={format})...")
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama

    kwargs = {"model": model_name, "temperature": temperature, "num_ctx": num_ctx, "base_url": base_url}
    if format:
        kwargs["format"] = format

    chat_instance = ChatOllama(**kwargs)
    _MODELS[cache_key] = chat_instance
    return chat_instance


# ==============================================================================
# 4. TOKENIZERS
# ==============================================================================
def get_embedding_tokenizer(model_name: Optional[str] = None, force_reload: bool = False):
    model_name = model_name or _get_config("EMBEDDING_TOKENIZER_NAME", "BAAI/bge-large-en-v1.5")
    return _load_tokenizer(model_name, force_reload)


def get_llm_tokenizer(model_name: Optional[str] = None, force_reload: bool = False):
    model_name = model_name or _get_config("LLM_TOKENIZER_NAME", "hf-internal-testing/llama-tokenizer")
    return _load_tokenizer(model_name, force_reload)


def get_splitter_tokenizer(model_name: Optional[str] = None, force_reload: bool = False):
    model_name = model_name or _get_config("MINILM_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    return _load_tokenizer(model_name, force_reload)


# ==============================================================================
# 5. VISION LANGUAGE MODEL (VLM)
# ==============================================================================
def get_vlm_model(force_reload: bool = False) -> Tuple[Any, Any]:
    """Retrieves the Qwen2-VL model and processor. Returns: (model, processor) tuple."""
    from src.ingestion.vlm_model import initialize_model
    return initialize_model()


def get_vlm_processor() -> Any:
    """Retrieves the Qwen2-VL processor."""
    return get_vlm_model()[1]


# ==============================================================================
# 6. EAGER INITIALIZER (WARMUP ALL MODELS)
# ==============================================================================
def init_all_models(include_vlm: bool = False) -> Dict[str, Any]:
    """Pre-warms and initializes all primary models and tokenizers."""
    logger.info("[MODELS] Initializing all core pipeline models...")
    models = {
        "embedding_model": get_embedding_model(),
        "sparse_embeddings": get_sparse_embedding_model(),
        "reranker": get_reranker(),
        "llm": get_llm(),
        "embedding_tokenizer": get_embedding_tokenizer(),
        "llm_tokenizer": get_llm_tokenizer(),
        "splitter_tokenizer": get_splitter_tokenizer(),
    }
    
    if include_vlm:
        logger.info("[MODELS] Initializing Vision Language Model...")
        models["vlm_model"], models["vlm_processor"] = get_vlm_model()

    logger.info("[MODELS] All models initialized successfully.")
    return models


# ==============================================================================
# 7. LAZY MODULE ATTRIBUTES (PEP 562)
# ==============================================================================
_LAZY_RESOLVERS = {
    "embedding_model": get_embedding_model,
    "sparse_embeddings": get_sparse_embedding_model,
    "reranker": get_reranker,
    "llm": get_llm,
    "chat_llm": get_chat_llm,
    "tokenizer": get_embedding_tokenizer,
    "embedding_tokenizer": get_embedding_tokenizer,
    "llm_tokenizer": get_llm_tokenizer,
    "splitter_tokenizer": get_splitter_tokenizer,
    "vlm_model": lambda: get_vlm_model()[0],
    "vlm_processor": lambda: get_vlm_model()[1],
}

def __getattr__(name: str) -> Any:
    """Enables direct module-level imports (e.g. `from src.common.models import llm`)."""
    if name in _LAZY_RESOLVERS:
        resolved = _LAZY_RESOLVERS[name]()
        globals()[name] = resolved
        return resolved

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Factory & getter functions
    "get_embedding_model", "get_sparse_embedding_model", "get_reranker", 
    "get_llm", "get_chat_llm", "get_embedding_tokenizer", "get_llm_tokenizer", 
    "get_splitter_tokenizer", "get_vlm_model", "get_vlm_processor", "init_all_models",
    
    # Pre-bound lazy instances
    "embedding_model", "sparse_embeddings", "reranker", "llm", "chat_llm", 
    "tokenizer", "embedding_tokenizer", "llm_tokenizer", "splitter_tokenizer",
]