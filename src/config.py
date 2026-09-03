"""Central configuration.

Every tunable the README calls out as "a parameter, not a hardcoded magic
number" (chunk size/overlap, top-k, RRF k, rerank top-N) lives here, each
overridable via an environment variable so behavior can change per
deployment without touching code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# --- Chunking (Step 1) ---
CHUNK_SIZE_TOKENS = _int_env("CHUNK_SIZE_TOKENS", 400)
CHUNK_OVERLAP_TOKENS = _int_env("CHUNK_OVERLAP_TOKENS", 50)

# --- Retrieval top-k (Steps 2, 4, 9) ---
DEFAULT_TOP_K = _int_env("DEFAULT_TOP_K", 5)

# --- Hybrid retrieval (Steps 6-8) ---
RRF_K = _int_env("RRF_K", 60)
FUSION_CANDIDATE_POOL = _int_env("FUSION_CANDIDATE_POOL", 30)  # how many each of dense/sparse feed into RRF
RERANK_TOP_N = _int_env("RERANK_TOP_N", 20)  # how many fused candidates go to the cross-encoder

# --- Dense embeddings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = _int_env("EMBEDDING_DIM", 384)  # bge-small-en-v1.5 output dimension

# --- Cross-encoder reranker (Step 8) ---
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# --- Qdrant vector store (Step 2) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")

# --- Generation LLM (Step 3) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = _float_env("OLLAMA_TIMEOUT_SECONDS", 60.0)

# --- Shared chunk store (Step 1) ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
