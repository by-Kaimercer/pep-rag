import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw_peps" / "peps"
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "evals"
RESULTS_DIR = ROOT / "results"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CACHE_PATH = DATA_DIR / "llm_cache.sqlite"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("PEP_RAG_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("PEP_RAG_CHAT_MODEL", "qwen2.5:7b")

CHUNK_WORDS = 220  # target chunk size; sections are packed paragraph by paragraph
TOP_K = 5
