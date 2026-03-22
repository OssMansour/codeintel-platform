"""
CodeIntel Benchmarks — End-to-End Latency Profiler

Measures timing for:
    1. Embedding a query
    2. Qdrant hybrid search (per-collection + cross-collection)
    3. Cross-encoder reranking
    4. Full agent query round-trip
    5. Wiki doc generation (single module, if repo available)

All measurements are taken N times to produce percentile statistics.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from benchmarks.config import (
    BenchmarkSettings,
    LatencyBenchmarkResult,
    LatencyProfile,
    now_iso,
)

log = structlog.get_logger(__name__)

DEFAULT_ITERATIONS = 5

# Sample queries for latency benchmarking
_SAMPLE_QUERIES = [
    "How does the authentication middleware handle JWT tokens?",
    "What error handling patterns are used in the API layer?",
    "Show me the database connection pooling implementation",
    "Explain the caching strategy for user sessions",
    "What happens when a webhook is received from GitLab?",
]


# ---------------------------------------------------------------------------
# Individual profilers
# ---------------------------------------------------------------------------


def _profile_embedding(
    embedder: Any, queries: list[str], iterations: int
) -> LatencyProfile:
    """Profile embed_query latency."""
    profile = LatencyProfile(operation="embed_query")
    for _ in range(iterations):
        for q in queries:
            t0 = time.perf_counter()
            embedder.embed_query(q)
            profile.samples.append((time.perf_counter() - t0) * 1000)
    return profile


def _profile_search(
    qdrant: Any,
    embedder: Any,
    queries: list[str],
    collection: str,
    iterations: int,
) -> LatencyProfile:
    """Profile single-collection hybrid search latency."""
    profile = LatencyProfile(operation=f"search_{collection}")
    for _ in range(iterations):
        for q in queries:
            vec = embedder.embed_query(q)
            t0 = time.perf_counter()
            qdrant.hybrid_search(
                collection=collection,
                query_vector=vec,
                query_text=q,
                top_k=20,
            )
            profile.samples.append((time.perf_counter() - t0) * 1000)
    return profile


def _profile_cross_collection_search(
    qdrant: Any, embedder: Any, queries: list[str], iterations: int
) -> LatencyProfile:
    """Profile search_all_collections latency."""
    profile = LatencyProfile(operation="search_all_collections")
    for _ in range(iterations):
        for q in queries:
            vec = embedder.embed_query(q)
            t0 = time.perf_counter()
            qdrant.search_all_collections(
                query_vector=vec,
                query_text=q,
                top_k_per_collection=10,
            )
            profile.samples.append((time.perf_counter() - t0) * 1000)
    return profile


def _profile_reranking(
    reranker: Any,
    qdrant: Any,
    embedder: Any,
    queries: list[str],
    iterations: int,
) -> LatencyProfile:
    """Profile cross-encoder reranking latency (search + rerank)."""
    profile = LatencyProfile(operation="rerank")
    for _ in range(iterations):
        for q in queries:
            vec = embedder.embed_query(q)
            results = qdrant.search_all_collections(
                query_vector=vec,
                query_text=q,
                top_k_per_collection=10,
            )
            if not results:
                continue
            t0 = time.perf_counter()
            reranker.rerank(q, results, top_k=5)
            profile.samples.append((time.perf_counter() - t0) * 1000)
    return profile


def _profile_agent_roundtrip(
    agent_api_url: str, queries: list[str], iterations: int
) -> LatencyProfile:
    """Profile full agent query round-trip via HTTP."""
    import httpx

    profile = LatencyProfile(operation="agent_roundtrip")
    for _ in range(iterations):
        for q in queries:
            try:
                t0 = time.perf_counter()
                resp = httpx.post(
                    f"{agent_api_url}/query",
                    json={"query": q, "stream": False},
                    timeout=120.0,
                )
                resp.raise_for_status()
                profile.samples.append((time.perf_counter() - t0) * 1000)
            except Exception as exc:
                log.warning(
                    "latency.agent_error", query=q[:60], error=str(exc)
                )
    return profile


# ---------------------------------------------------------------------------
# Full latency benchmark
# ---------------------------------------------------------------------------


def run_latency_benchmark(
    settings: BenchmarkSettings | None = None,
    queries: list[str] | None = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> LatencyBenchmarkResult:
    """
    Run all latency profiling suites.

    Parameters
    ----------
    settings : BenchmarkSettings
    queries : Custom queries (defaults to built-in sample set)
    iterations : Number of times to repeat each query
    """
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_qdrant_store, QdrantSettings

    settings = settings or BenchmarkSettings()
    queries = queries or _SAMPLE_QUERIES[:3]  # Keep it manageable

    log.info(
        "latency_benchmark.start",
        n_queries=len(queries),
        iterations=iterations,
    )
    t_start = time.perf_counter()

    embedder = get_embedder()
    qdrant = get_qdrant_store()
    qs = QdrantSettings()

    profiles: dict[str, LatencyProfile] = {}

    # 1. Embedding
    log.info("latency_benchmark.profiling", operation="embed_query")
    profiles["embed_query"] = _profile_embedding(embedder, queries, iterations)

    # 2. Per-collection search
    for coll in [qs.qdrant_collection_code, qs.qdrant_collection_app_docs]:
        try:
            if qdrant.is_collection_populated(coll):
                log.info("latency_benchmark.profiling", operation=f"search_{coll}")
                profiles[f"search_{coll}"] = _profile_search(
                    qdrant, embedder, queries, coll, iterations
                )
        except Exception as exc:
            log.warning("latency_benchmark.collection_skip", collection=coll, error=str(exc))

    # 3. Cross-collection search
    log.info("latency_benchmark.profiling", operation="search_all_collections")
    profiles["search_all_collections"] = _profile_cross_collection_search(
        qdrant, embedder, queries, iterations
    )

    # 4. Reranking
    try:
        from services.agent.reranker import get_reranker

        reranker = get_reranker()
        log.info("latency_benchmark.profiling", operation="rerank")
        profiles["rerank"] = _profile_reranking(
            reranker, qdrant, embedder, queries, iterations
        )
    except Exception as exc:
        log.warning("latency_benchmark.reranker_skip", error=str(exc))

    # 5. Full agent round-trip (only if agent is reachable)
    try:
        import httpx

        resp = httpx.get(f"{settings.agent_api_url}/health", timeout=5.0)
        if resp.status_code == 200:
            log.info("latency_benchmark.profiling", operation="agent_roundtrip")
            profiles["agent_roundtrip"] = _profile_agent_roundtrip(
                settings.agent_api_url, queries[:2], max(1, iterations // 2)
            )
    except Exception:
        log.info("latency_benchmark.agent_unreachable")

    duration_s = time.perf_counter() - t_start

    # Log summary
    for name, p in profiles.items():
        log.info(
            "latency_benchmark.profile",
            operation=name,
            samples=len(p.samples),
            mean_ms=f"{p.mean_ms:.1f}",
            p95_ms=f"{p.p95_ms:.1f}",
        )

    return LatencyBenchmarkResult(
        profiles=profiles,
        timestamp=now_iso(),
        duration_s=duration_s,
    )
