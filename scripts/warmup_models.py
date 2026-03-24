#!/usr/bin/env python3
"""
Pre-download and cache both ML models so the first agent query isn't blocked
waiting for a 550MB download.

Usage:
    conda run -n codeintel python scripts/warmup_models.py
"""
import os
import sys
import time
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

EMBED_CACHE    = os.environ.get("EMBED_CACHE_DIR",    "C:/codeintel-data/models/embed")
RERANKER_CACHE = os.environ.get("RERANKER_CACHE_DIR", "C:/codeintel-data/models/reranker")
EMBED_MODEL    = os.environ.get("EMBED_MODEL",        "nomic-ai/CodeRankEmbed")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL",     "cross-encoder/ms-marco-MiniLM-L6-v2")

Path(EMBED_CACHE).mkdir(parents=True, exist_ok=True)
Path(RERANKER_CACHE).mkdir(parents=True, exist_ok=True)

# ── Reranker (~34 MB) ─────────────────────────────────────────────────────
print(f"\n[1/2] Downloading reranker: {RERANKER_MODEL}")
t = time.time()
from sentence_transformers import CrossEncoder
reranker = CrossEncoder(RERANKER_MODEL, device="cpu", cache_dir=RERANKER_CACHE)
# Run a tiny prediction to ensure it's fully initialised
reranker.predict([("test query", "test document")])
print(f"      ✓ Reranker ready in {time.time()-t:.1f}s")

# ── Embedder (~550 MB) ────────────────────────────────────────────────────
print(f"\n[2/2] Downloading embedder: {EMBED_MODEL}  (this may take a while)")
t = time.time()
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(
    EMBED_MODEL,
    device="cpu",
    cache_folder=EMBED_CACHE,
    trust_remote_code=True,
)
vec = embedder.encode(["warmup"], normalize_embeddings=True)
print(f"      ✓ Embedder ready in {time.time()-t:.1f}s — dim={vec.shape[1]}")

print("\n✅  Both models cached. Agent queries will be fast on next start.\n")
