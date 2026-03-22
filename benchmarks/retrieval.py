"""
CodeIntel Benchmarks — Retrieval Quality Evaluation

Measures retrieval precision/recall using MRR, Hit@K, and nDCG.
Runs queries against Qdrant + reranker and checks whether expected
files/symbols appear in the returned results.
"""

from __future__ import annotations

import math
import time
from typing import Any

import structlog

from benchmarks.config import (
    BenchmarkSettings,
    RetrievalBenchmarkResult,
    RetrievalCaseResult,
    RetrievalTestCase,
    now_iso,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _reciprocal_rank(ranked_items: list[bool]) -> float:
    """Reciprocal rank: 1/position of first relevant hit (0 if none)."""
    for i, hit in enumerate(ranked_items, 1):
        if hit:
            return 1.0 / i
    return 0.0


def _hit_at_k(ranked_items: list[bool], k: int) -> bool:
    """Whether any relevant item appears in the top-k results."""
    return any(ranked_items[:k])


def _ndcg_at_k(ranked_items: list[bool], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ k.

    Binary relevance: relevant items get score 1, others 0.
    Ideal DCG assumes all relevant items are at the top.
    """
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, rel in enumerate(ranked_items[:k]) if rel
    )
    # Ideal DCG: all relevant items packed at the top
    n_relevant = sum(ranked_items[:k])
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Result relevance checker
# ---------------------------------------------------------------------------


def _is_result_relevant(
    result: dict[str, Any],
    expected_files: list[str],
    expected_symbols: list[str],
) -> bool:
    """
    Check if a single search result matches any expected file or symbol.

    Uses substring matching so partial paths work (e.g. "auth/handler.py"
    matches "/data/repos/project/auth/handler.py").
    """
    file_path = (
        result.get("file_path", "")
        or result.get("metadata", {}).get("file_path", "")
    ).replace("\\", "/")

    symbol_name = (
        result.get("symbol_name", "")
        or result.get("metadata", {}).get("symbol_name", "")
    )

    for ef in expected_files:
        ef_norm = ef.replace("\\", "/")
        if ef_norm and (ef_norm in file_path or file_path.endswith(ef_norm)):
            return True

    for es in expected_symbols:
        if es and es == symbol_name:
            return True

    return False


# ---------------------------------------------------------------------------
# Single-case evaluator
# ---------------------------------------------------------------------------


def evaluate_retrieval_case(
    case: RetrievalTestCase,
    embedder: Any,
    qdrant: Any,
    reranker: Any | None = None,
    top_k: int = 20,
) -> RetrievalCaseResult:
    """
    Run one retrieval test case and compute all metrics.

    Parameters
    ----------
    case : RetrievalTestCase
    embedder : CodeEmbedder instance
    qdrant : QdrantStore instance
    reranker : CrossEncoderReranker instance (optional)
    top_k : Number of results to retrieve
    """
    t0 = time.perf_counter()

    # Embed query
    query_vec = embedder.embed_query(case.query)

    # Build filter
    filters = {}
    if case.project_id:
        filters["project_id"] = case.project_id

    # Search: hybrid across all collections, or scoped to one
    if case.expected_collection:
        raw_results = qdrant.hybrid_search(
            collection=case.expected_collection,
            query_vector=query_vec,
            query_text=case.query,
            filters=filters,
            top_k=top_k,
        )
    else:
        raw_results = qdrant.search_all_collections(
            query_vector=query_vec,
            query_text=case.query,
            filters=filters,
            top_k_per_collection=top_k,
        )

    # Optional reranking
    if reranker and raw_results:
        ranked = reranker.rerank(case.query, raw_results, top_k=top_k)
        results = [r.chunk for r in ranked]
    else:
        results = raw_results

    latency_ms = (time.perf_counter() - t0) * 1000

    # Compute relevance vector
    expected_files = case.expected_files
    expected_symbols = case.expected_symbols
    relevance = [
        _is_result_relevant(r, expected_files, expected_symbols) for r in results
    ]

    # Gather matched items
    matched_files = []
    matched_symbols = []
    for r, rel in zip(results, relevance):
        if rel:
            fp = r.get("file_path", r.get("metadata", {}).get("file_path", ""))
            sn = r.get("symbol_name", r.get("metadata", {}).get("symbol_name", ""))
            if fp and fp not in matched_files:
                matched_files.append(fp)
            if sn and sn not in matched_symbols:
                matched_symbols.append(sn)

    # Top results for debugging
    top_results = []
    for i, r in enumerate(results[:5]):
        top_results.append(
            {
                "rank": i + 1,
                "file_path": r.get("file_path", r.get("metadata", {}).get("file_path", "")),
                "symbol_name": r.get("symbol_name", r.get("metadata", {}).get("symbol_name", "")),
                "score": r.get("score", 0.0),
                "relevant": relevance[i] if i < len(relevance) else False,
            }
        )

    rr = _reciprocal_rank(relevance)

    return RetrievalCaseResult(
        query_id=case.query_id,
        query=case.query,
        reciprocal_rank=rr,
        hit_at_1=_hit_at_k(relevance, 1),
        hit_at_3=_hit_at_k(relevance, 3),
        hit_at_5=_hit_at_k(relevance, 5),
        hit_at_10=_hit_at_k(relevance, 10),
        ndcg_at_5=_ndcg_at_k(relevance, 5),
        ndcg_at_10=_ndcg_at_k(relevance, 10),
        latency_ms=latency_ms,
        result_count=len(results),
        matched_files=matched_files,
        matched_symbols=matched_symbols,
        top_results=top_results,
        passed=(rr > 0),
    )


# ---------------------------------------------------------------------------
# Full retrieval benchmark
# ---------------------------------------------------------------------------


def run_retrieval_benchmark(
    cases: list[RetrievalTestCase],
    settings: BenchmarkSettings | None = None,
) -> RetrievalBenchmarkResult:
    """
    Run all retrieval test cases and aggregate metrics.

    Lazily imports embedder/qdrant/reranker to avoid import-time overhead.
    """
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_qdrant_store
    from services.agent.reranker import get_reranker

    settings = settings or BenchmarkSettings()

    log.info("retrieval_benchmark.start", total_cases=len(cases))
    t_start = time.perf_counter()

    embedder = get_embedder()
    qdrant = get_qdrant_store()

    try:
        reranker = get_reranker()
    except Exception:
        log.warning("retrieval_benchmark.reranker_unavailable")
        reranker = None

    case_results: list[RetrievalCaseResult] = []
    for case in cases:
        try:
            result = evaluate_retrieval_case(case, embedder, qdrant, reranker)
            case_results.append(result)
            log.info(
                "retrieval_benchmark.case_done",
                query_id=case.query_id,
                mrr=f"{result.reciprocal_rank:.3f}",
                hit_at_5=result.hit_at_5,
                latency_ms=f"{result.latency_ms:.0f}",
            )
        except Exception as exc:
            log.error("retrieval_benchmark.case_error", query_id=case.query_id, error=str(exc))
            case_results.append(
                RetrievalCaseResult(
                    query_id=case.query_id,
                    query=case.query,
                    passed=False,
                    failure_reason=str(exc),
                )
            )

    duration_s = time.perf_counter() - t_start

    # Aggregate
    n = len(case_results) or 1
    latencies = [c.latency_ms for c in case_results if c.latency_ms > 0]
    sorted_lat = sorted(latencies) if latencies else [0.0]
    p95_idx = int(len(sorted_lat) * 0.95)

    return RetrievalBenchmarkResult(
        mean_mrr=sum(c.reciprocal_rank for c in case_results) / n,
        mean_hit_at_1=sum(c.hit_at_1 for c in case_results) / n,
        mean_hit_at_3=sum(c.hit_at_3 for c in case_results) / n,
        mean_hit_at_5=sum(c.hit_at_5 for c in case_results) / n,
        mean_hit_at_10=sum(c.hit_at_10 for c in case_results) / n,
        mean_ndcg_at_5=sum(c.ndcg_at_5 for c in case_results) / n,
        mean_ndcg_at_10=sum(c.ndcg_at_10 for c in case_results) / n,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency_ms=sorted_lat[min(p95_idx, len(sorted_lat) - 1)],
        total_cases=len(case_results),
        passed_cases=sum(1 for c in case_results if c.passed),
        pass_rate=sum(1 for c in case_results if c.passed) / n,
        case_results=case_results,
        timestamp=now_iso(),
        duration_s=duration_s,
    )
