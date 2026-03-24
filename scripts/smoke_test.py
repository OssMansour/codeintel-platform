#!/usr/bin/env python3
"""
Quick smoke test: runs a single agent query end-to-end without HTTP.
Use this to diagnose timing issues before containerising.

Usage:
    conda run -n codeintel python scripts/smoke_test.py
"""
import os, sys, time
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Override Docker hostnames with localhost for local testing
for k, v in os.environ.items():
    if "redis" in v.lower() and "redis:" in v:
        os.environ[k] = v.replace("redis:", "localhost:")
    if "qdrant" in v.lower() and "qdrant:" in v:
        os.environ[k] = v.replace("qdrant:", "localhost:")
    if "ollama" in v.lower() and "ollama:" in v:
        os.environ[k] = v.replace("ollama:", "localhost:")

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG
    logger_factory=structlog.PrintLoggerFactory(),
)

print("=" * 60)
print("CodeIntel Smoke Test")
print("=" * 60)

# ── 1. Qdrant connectivity ────────────────────────────────────
print("\n[1] Qdrant connection...")
t = time.time()
from services.indexing.qdrant_store import get_store
store = get_store()
stats = store.get_all_collection_stats()
for name, s in stats.items():
    print(f"    {name}: {s['points_count']} points  ({s['status']})")
print(f"    ✓ {time.time()-t:.1f}s")

# ── 2. Embedder ───────────────────────────────────────────────
print("\n[2] Loading embedder...")
t = time.time()
from services.indexing.embedder import get_embedder
emb = get_embedder()
vecs = emb.embed_texts(["def main(): pass"], use_query_prefix=False)
print(f"    ✓ dim={len(vecs[0])}  {time.time()-t:.1f}s")

# ── 3. Reranker ───────────────────────────────────────────────
print("\n[3] Loading reranker...")
t = time.time()
from services.agent.reranker import get_reranker
rr = get_reranker()
chunk = {"content": "def main(): pass", "chunk_id": "x", "score": 0.9, "collection": "code_repo",
         "file_path": "main.py", "symbol_name": "main", "start_line": 1, "end_line": 1}
ranked = rr.rerank("what is the main entry point", [chunk], top_k=1)
print(f"    ✓ reranker score={ranked[0].score:.3f}  {time.time()-t:.1f}s")

# ── 4. Hybrid search ──────────────────────────────────────────
print("\n[4] Hybrid search (top-5)...")
t = time.time()
q_vec = emb.embed_query("main entry point planning algorithm")
results = store.hybrid_search("code_repo", q_vec, "main entry point", top_k=5)
print(f"    ✓ {len(results)} results  {time.time()-t:.1f}s")
for r in results[:3]:
    print(f"       {r['metadata'].get('file_path','?')}:{r['metadata'].get('start_line','?')}  score={r['score']:.3f}")

# ── 5. LLM (single call) ──────────────────────────────────────
print("\n[5] LLM call (single turn)...")
t = time.time()
from services.llm import get_provider
llm = get_provider()
from services.llm.base import ChatMessage
resp = llm.chat([
    ChatMessage("system", "You are a concise code assistant."),
    ChatMessage("user", "In one sentence: what does a 'main entry point' mean in Python?"),
])
print(f"    ✓ {time.time()-t:.1f}s")
print(f"    Response: {resp.content[:200]}")

# ── 6. Full agent query ───────────────────────────────────────
print("\n[6] Full ReAct agent query...")
t = time.time()
from services.agent.agent import CodeIntelAgent
agent = CodeIntelAgent()
agent._graph = agent._build_graph()
result = agent.query(
    question="What are the main entry points and how does the planning algorithm work?",
    project_id="OssMansour/DHP-Multi-Step-Planning",
)
elapsed = time.time() - t
print(f"    ✓ {elapsed:.1f}s")
print(f"    Confidence : {result.confidence:.2f}")
print(f"    Tools used : {result.tool_calls_made}")
print(f"    Sources    : {len(result.sources)}")
print(f"    Answer     : {result.answer[:400]}")

print("\n✅  Smoke test complete!\n")
