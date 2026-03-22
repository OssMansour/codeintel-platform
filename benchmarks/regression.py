"""
CodeIntel Benchmarks — Regression Test Harness

Runs agent queries and asserts:
    - Expected substrings appear in answers
    - Forbidden substrings do not appear
    - Minimum confidence thresholds are met
    - Expected tools are invoked
    - Latency stays within bounds

Designed for CI integration: exits with non-zero code on failures.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from benchmarks.config import (
    AgentCaseResult,
    AgentTestCase,
    BenchmarkSettings,
    RegressionResult,
    now_iso,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-case executor
# ---------------------------------------------------------------------------


def _run_agent_case(
    case: AgentTestCase,
    agent_api_url: str,
    global_max_latency: float,
) -> AgentCaseResult:
    """
    Execute a single agent query and validate against assertions.
    """
    import httpx

    failures: list[str] = []
    max_lat = case.max_latency_s if case.max_latency_s > 0 else global_max_latency

    # Execute query
    t0 = time.perf_counter()
    try:
        resp = httpx.post(
            f"{agent_api_url}/query",
            json={
                "query": case.query,
                "project_id": case.project_id,
                "stream": False,
            },
            timeout=max(max_lat * 2, 60.0),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return AgentCaseResult(
            query_id=case.query_id,
            query=case.query,
            passed=False,
            failures=[f"HTTP error: {exc}"],
            latency_s=time.perf_counter() - t0,
        )

    latency_s = time.perf_counter() - t0
    answer = data.get("answer", "")
    confidence = data.get("confidence", 0.0)
    tool_calls = data.get("tool_calls_made", [])
    collection_hits = data.get("collection_hits", {})

    # --- Assertions ---

    # 1. must_contain
    answer_lower = answer.lower()
    for expected in case.must_contain:
        if expected.lower() not in answer_lower:
            failures.append(f"MISSING: answer must contain '{expected}'")

    # 2. must_not_contain
    for forbidden in case.must_not_contain:
        if forbidden.lower() in answer_lower:
            failures.append(f"FORBIDDEN: answer contains '{forbidden}'")

    # 3. min_confidence
    if case.min_confidence > 0 and confidence < case.min_confidence:
        failures.append(
            f"CONFIDENCE: {confidence:.2f} < {case.min_confidence:.2f}"
        )

    # 4. expected_tool_calls
    for tool in case.expected_tool_calls:
        if tool not in tool_calls:
            failures.append(f"TOOL_MISSING: expected '{tool}' in tool calls")

    # 5. latency
    if max_lat > 0 and latency_s > max_lat:
        failures.append(
            f"LATENCY: {latency_s:.1f}s > {max_lat:.1f}s"
        )

    return AgentCaseResult(
        query_id=case.query_id,
        query=case.query,
        answer_snippet=answer[:300],
        confidence=confidence,
        tool_calls_made=tool_calls,
        collection_hits=collection_hits,
        latency_s=latency_s,
        passed=len(failures) == 0,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Full regression suite
# ---------------------------------------------------------------------------


def run_regression_tests(
    cases: list[AgentTestCase],
    settings: BenchmarkSettings | None = None,
) -> RegressionResult:
    """
    Run all agent regression test cases.

    Parameters
    ----------
    cases : list[AgentTestCase]
    settings : BenchmarkSettings

    Returns
    -------
    RegressionResult with pass/fail for each case.
    """
    settings = settings or BenchmarkSettings()

    log.info("regression.start", total_cases=len(cases))
    t_start = time.perf_counter()

    results: list[AgentCaseResult] = []
    for case in cases:
        try:
            result = _run_agent_case(
                case,
                settings.agent_api_url,
                settings.max_agent_latency_s,
            )
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            log.info(
                "regression.case_done",
                query_id=case.query_id,
                status=status,
                latency_s=f"{result.latency_s:.1f}",
                failures=result.failures[:3] if result.failures else [],
            )
        except Exception as exc:
            log.error("regression.case_error", query_id=case.query_id, error=str(exc))
            results.append(
                AgentCaseResult(
                    query_id=case.query_id,
                    query=case.query,
                    passed=False,
                    failures=[f"Unhandled error: {exc}"],
                )
            )

    duration_s = time.perf_counter() - t_start
    n_passed = sum(1 for r in results if r.passed)

    return RegressionResult(
        total_cases=len(results),
        passed_cases=n_passed,
        failed_cases=len(results) - n_passed,
        pass_rate=n_passed / len(results) if results else 0.0,
        agent_results=results,
        timestamp=now_iso(),
        duration_s=duration_s,
    )
