"""
CodeIntel Benchmarks — Wiki Documentation Quality Scorer

Evaluates generated wiki docs using the same DOC_CRITIQUE_PROMPT from Phase 3.
Produces per-doc and aggregate quality scores across 5 criteria:
    completeness, accuracy, diagrams, clarity, cross_references (each 1–5).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import structlog

from benchmarks.config import (
    BenchmarkSettings,
    WikiQualityCaseResult,
    WikiQualityResult,
    WikiQualityTestCase,
    now_iso,
)

log = structlog.get_logger(__name__)

_REVIEW_RE = re.compile(r"<REVIEW>\s*(.*?)\s*</REVIEW>", re.DOTALL)

_CRITERIA = ["completeness", "accuracy", "diagrams", "clarity", "cross_references"]


# ---------------------------------------------------------------------------
# Parse review JSON
# ---------------------------------------------------------------------------


def _parse_review(raw: str) -> dict[str, Any]:
    """
    Extract the JSON review block from LLM output.

    Expects format wrapped in <REVIEW>...</REVIEW> tags.
    Falls back to raw JSON parse if tags are absent.
    """
    m = _REVIEW_RE.search(raw)
    text = m.group(1) if m else raw.strip()

    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("wiki_quality.json_parse_failed", raw_snippet=text[:200])
        return {}


# ---------------------------------------------------------------------------
# Score a single document
# ---------------------------------------------------------------------------


def score_document(
    module_name: str,
    doc_content: str,
    llm: Any,
) -> WikiQualityCaseResult:
    """
    Run the critique prompt against a single wiki document.

    Uses the same DOC_CRITIQUE_PROMPT that the refiner uses,
    but purely as an evaluation step (no refinement loop).
    """
    from services.docgen.prompts import DOC_CRITIQUE_PROMPT
    from services.llm.base import ChatMessage

    t0 = time.perf_counter()

    prompt = DOC_CRITIQUE_PROMPT.format(
        module_name=module_name,
        document=doc_content,
    )

    try:
        response = llm.complete(prompt, max_tokens=1024, temperature=0.1)
        raw_text = response.text
    except Exception as exc:
        log.error("wiki_quality.llm_error", module=module_name, error=str(exc))
        return WikiQualityCaseResult(
            project_id="",
            module_name=module_name,
            scores={c: 0 for c in _CRITERIA},
            overall=0.0,
            issues=[f"LLM error: {exc}"],
            passed=False,
            doc_length_chars=len(doc_content),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    review = _parse_review(raw_text)
    scores = {}
    for c in _CRITERIA:
        val = review.get(c, 0)
        scores[c] = int(val) if isinstance(val, (int, float)) else 0

    issues = review.get("issues", [])
    if isinstance(issues, str):
        issues = [issues]

    overall = sum(scores.values()) / len(scores) if scores else 0.0
    passed = overall >= 3.5 and all(v >= 2 for v in scores.values())

    latency_ms = (time.perf_counter() - t0) * 1000

    return WikiQualityCaseResult(
        project_id="",
        module_name=module_name,
        scores=scores,
        overall=round(overall, 2),
        issues=issues,
        passed=passed,
        doc_length_chars=len(doc_content),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Load wiki docs from disk
# ---------------------------------------------------------------------------


def _discover_wiki_docs(
    wiki_base: str | Path, project_id: str
) -> list[tuple[str, str]]:
    """
    Discover all wiki markdown docs for a project.

    Returns list of (module_name, content) tuples.
    """
    project_dir = Path(wiki_base) / project_id
    if not project_dir.is_dir():
        log.warning("wiki_quality.project_dir_missing", path=str(project_dir))
        return []

    docs = []
    for md_file in sorted(project_dir.glob("*.md")):
        if md_file.name == "overview.md":
            continue  # skip overview, evaluate module docs only
        module_name = md_file.stem
        content = md_file.read_text(encoding="utf-8", errors="replace")
        if content.strip():
            docs.append((module_name, content))

    return docs


# ---------------------------------------------------------------------------
# Full wiki quality benchmark
# ---------------------------------------------------------------------------


def run_wiki_quality_benchmark(
    cases: list[WikiQualityTestCase],
    settings: BenchmarkSettings | None = None,
) -> WikiQualityResult:
    """
    Evaluate wiki documentation quality for all specified projects/modules.

    If a case specifies a module_name, only that module is scored.
    Otherwise, all modules for the project are sampled.
    """
    from services.llm import get_provider

    settings = settings or BenchmarkSettings()
    wiki_base = settings.wiki_docs_dir

    log.info("wiki_quality_benchmark.start", total_cases=len(cases))
    t_start = time.perf_counter()

    llm = get_provider()
    case_results: list[WikiQualityCaseResult] = []

    for case in cases:
        if case.module_name:
            # Score a specific module
            doc_path = Path(wiki_base) / case.project_id / f"{case.module_name}.md"
            if not doc_path.is_file():
                log.warning(
                    "wiki_quality.doc_not_found",
                    project=case.project_id,
                    module=case.module_name,
                )
                case_results.append(
                    WikiQualityCaseResult(
                        project_id=case.project_id,
                        module_name=case.module_name,
                        passed=False,
                        issues=["Document file not found"],
                    )
                )
                continue

            content = doc_path.read_text(encoding="utf-8", errors="replace")
            result = score_document(case.module_name, content, llm)
            result.project_id = case.project_id
            result.passed = result.passed and result.overall >= case.min_overall
            case_results.append(result)
        else:
            # Score all modules in project
            docs = _discover_wiki_docs(wiki_base, case.project_id)
            if not docs:
                log.warning("wiki_quality.no_docs", project=case.project_id)
                continue

            for module_name, content in docs:
                result = score_document(module_name, content, llm)
                result.project_id = case.project_id
                result.passed = result.passed and result.overall >= case.min_overall
                case_results.append(result)

                log.info(
                    "wiki_quality.doc_scored",
                    project=case.project_id,
                    module=module_name,
                    overall=result.overall,
                    passed=result.passed,
                )

    duration_s = time.perf_counter() - t_start
    n = len(case_results) or 1

    # Aggregate per-criterion means
    def _mean(criterion: str) -> float:
        vals = [c.scores.get(criterion, 0) for c in case_results]
        return sum(vals) / len(vals) if vals else 0.0

    return WikiQualityResult(
        mean_overall=sum(c.overall for c in case_results) / n,
        mean_completeness=_mean("completeness"),
        mean_accuracy=_mean("accuracy"),
        mean_diagrams=_mean("diagrams"),
        mean_clarity=_mean("clarity"),
        mean_cross_refs=_mean("cross_references"),
        total_docs=len(case_results),
        passed_docs=sum(1 for c in case_results if c.passed),
        pass_rate=sum(1 for c in case_results if c.passed) / n,
        case_results=case_results,
        timestamp=now_iso(),
        duration_s=duration_s,
    )
