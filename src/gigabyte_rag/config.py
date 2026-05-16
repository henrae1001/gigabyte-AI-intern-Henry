from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "indexes"
MODEL_DIR = ROOT / "models"

RAW_HTML_PATH = RAW_DIR / "aorus_master_16_am6h.html"
SPECS_JSON_PATH = PROCESSED_DIR / "specs.json"
SPECS_MD_PATH = PROCESSED_DIR / "specs.md"
CHUNKS_JSON_PATH = PROCESSED_DIR / "chunks.json"
INDEX_PATH = INDEX_DIR / "vectors.json"
DEFAULT_MODEL_PATH = MODEL_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
