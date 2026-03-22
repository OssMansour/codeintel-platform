"""
CodeIntel Benchmarks — Report Generator

Produces two output formats:
    1. **JSON** — Machine-readable for CI/CD, dashboards, and trend tracking
    2. **Markdown** — Human-readable summary with tables and pass/fail badges

Reports are written to BENCH_OUTPUT_DIR (default: benchmarks/results/).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from benchmarks.config import (
    BenchmarkSettings,
    FullBenchmarkReport,
    LatencyBenchmarkResult,
    LatencyProfile,
    RetrievalBenchmarkResult,
    RegressionResult,
    WikiQualityResult,
    ensure_dir,
    now_iso,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """Recursively convert dataclass trees to JSON-safe dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        d = {}
        for k in obj.__dataclass_fields__:
            d[k] = _to_serializable(getattr(obj, k))
        return d
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def write_json_report(report: FullBenchmarkReport, output_dir: str) -> Path:
    """Write full benchmark report as a timestamped JSON file."""
    out = ensure_dir(output_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = out / f"benchmark_{ts}.json"

    data = _to_serializable(report)
    filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    log.info("report.json_written", path=str(filepath))
    return filepath


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _badge(passed: bool) -> str:
    return "✅" if passed else "❌"


def _pf(val: float, fmt: str = ".2f") -> str:
    return f"{val:{fmt}}"


def write_markdown_report(
    report: FullBenchmarkReport,
    output_dir: str,
    settings: BenchmarkSettings | None = None,
) -> Path:
    """Write a human-readable Markdown benchmark report."""
    settings = settings or BenchmarkSettings()
    out = ensure_dir(output_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = out / f"benchmark_{ts}.md"

    lines: list[str] = []
    _l = lines.append

    _l("# CodeIntel Benchmark Report")
    _l(f"\n**Generated**: {report.timestamp}")
    _l(f"**Total Duration**: {report.total_duration_s:.1f}s")
    _l("")

    # ----- Retrieval -----
    if report.retrieval:
        r = report.retrieval
        _l("---")
        _l("## 📊 Retrieval Quality")
        _l("")
        _l("| Metric | Value | Threshold | Status |")
        _l("|--------|-------|-----------|--------|")
        _l(f"| MRR | {_pf(r.mean_mrr)} | ≥ {settings.min_mrr} | {_badge(r.mean_mrr >= settings.min_mrr)} |")
        _l(f"| Hit@3 | {_pf(r.mean_hit_at_3)} | ≥ {settings.min_hit_at_3} | {_badge(r.mean_hit_at_3 >= settings.min_hit_at_3)} |")
        _l(f"| Hit@5 | {_pf(r.mean_hit_at_5)} | ≥ {settings.min_hit_at_5} | {_badge(r.mean_hit_at_5 >= settings.min_hit_at_5)} |")
        _l(f"| nDCG@5 | {_pf(r.mean_ndcg_at_5)} | ≥ {settings.min_ndcg_at_5} | {_badge(r.mean_ndcg_at_5 >= settings.min_ndcg_at_5)} |")
        _l(f"| Hit@1 | {_pf(r.mean_hit_at_1)} | — | ℹ️ |")
        _l(f"| Hit@10 | {_pf(r.mean_hit_at_10)} | — | ℹ️ |")
        _l(f"| nDCG@10 | {_pf(r.mean_ndcg_at_10)} | — | ℹ️ |")
        _l(f"| Mean Latency | {_pf(r.mean_latency_ms, '.0f')}ms | — | ℹ️ |")
        _l(f"| P95 Latency | {_pf(r.p95_latency_ms, '.0f')}ms | — | ℹ️ |")
        _l(f"| Cases | {r.passed_cases}/{r.total_cases} passed | — | {_badge(r.pass_rate >= 0.8)} |")
        _l("")

        # Per-case details (collapsed)
        if r.case_results:
            _l("<details>")
            _l("<summary>Per-Query Details</summary>")
            _l("")
            _l("| Query ID | MRR | Hit@5 | nDCG@5 | Latency | Status |")
            _l("|----------|-----|-------|--------|---------|--------|")
            for c in r.case_results:
                _l(
                    f"| {c.query_id} | {_pf(c.reciprocal_rank)} | "
                    f"{_badge(c.hit_at_5)} | {_pf(c.ndcg_at_5)} | "
                    f"{c.latency_ms:.0f}ms | {_badge(c.passed)} |"
                )
            _l("")
            _l("</details>")
            _l("")

    # ----- Wiki Quality -----
    if report.wiki_quality:
        w = report.wiki_quality
        _l("---")
        _l("## 📝 Wiki Documentation Quality")
        _l("")
        _l("| Criterion | Mean Score | Status |")
        _l("|-----------|-----------|--------|")
        _l(f"| **Overall** | {_pf(w.mean_overall)} / 5.0 | {_badge(w.mean_overall >= settings.min_wiki_quality)} |")
        _l(f"| Completeness | {_pf(w.mean_completeness)} | {_badge(w.mean_completeness >= 3)} |")
        _l(f"| Accuracy | {_pf(w.mean_accuracy)} | {_badge(w.mean_accuracy >= 3)} |")
        _l(f"| Diagrams | {_pf(w.mean_diagrams)} | {_badge(w.mean_diagrams >= 3)} |")
        _l(f"| Clarity | {_pf(w.mean_clarity)} | {_badge(w.mean_clarity >= 3)} |")
        _l(f"| Cross-References | {_pf(w.mean_cross_refs)} | {_badge(w.mean_cross_refs >= 3)} |")
        _l(f"| Docs Evaluated | {w.passed_docs}/{w.total_docs} passed | {_badge(w.pass_rate >= 0.7)} |")
        _l("")

        if w.case_results:
            _l("<details>")
            _l("<summary>Per-Document Scores</summary>")
            _l("")
            _l("| Project | Module | Overall | Comp | Acc | Diag | Clarity | XRef | Status |")
            _l("|---------|--------|---------|------|-----|------|---------|------|--------|")
            for c in w.case_results:
                s = c.scores
                _l(
                    f"| {c.project_id} | {c.module_name} | {_pf(c.overall)} | "
                    f"{s.get('completeness', 0)} | {s.get('accuracy', 0)} | "
                    f"{s.get('diagrams', 0)} | {s.get('clarity', 0)} | "
                    f"{s.get('cross_references', 0)} | {_badge(c.passed)} |"
                )
            _l("")
            _l("</details>")
            _l("")

    # ----- Latency -----
    if report.latency:
        lat = report.latency
        _l("---")
        _l("## ⏱️ Latency Profile")
        _l("")
        _l("| Operation | Samples | Mean | P50 | P95 | P99 | Min | Max |")
        _l("|-----------|---------|------|-----|-----|-----|-----|-----|")
        for name, p in lat.profiles.items():
            _l(
                f"| {name} | {len(p.samples)} | "
                f"{p.mean_ms:.1f}ms | {p.p50_ms:.1f}ms | "
                f"{p.p95_ms:.1f}ms | {p.p99_ms:.1f}ms | "
                f"{p.min_ms:.1f}ms | {p.max_ms:.1f}ms |"
            )
        _l("")

    # ----- Regression -----
    if report.regression:
        reg = report.regression
        _l("---")
        _l("## 🧪 Regression Tests")
        _l("")
        overall_pass = reg.failed_cases == 0
        _l(f"**Result**: {reg.passed_cases}/{reg.total_cases} passed "
           f"({reg.pass_rate:.0%}) {_badge(overall_pass)}")
        _l("")

        if reg.agent_results:
            _l("| Query ID | Confidence | Tools | Latency | Status | Failures |")
            _l("|----------|-----------|-------|---------|--------|----------|")
            for c in reg.agent_results:
                fail_str = "; ".join(c.failures[:2]) if c.failures else "—"
                _l(
                    f"| {c.query_id} | {c.confidence:.2f} | "
                    f"{', '.join(c.tool_calls_made[:3])} | "
                    f"{c.latency_s:.1f}s | {_badge(c.passed)} | {fail_str} |"
                )
            _l("")

    # ----- Footer -----
    _l("---")
    _l(f"*Report generated by CodeIntel Benchmark Suite v1.0*")
    _l("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    log.info("report.markdown_written", path=str(filepath))
    return filepath


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_reports(
    report: FullBenchmarkReport,
    settings: BenchmarkSettings | None = None,
) -> tuple[Path, Path]:
    """
    Generate both JSON and Markdown reports.

    Returns (json_path, markdown_path).
    """
    settings = settings or BenchmarkSettings()
    output_dir = settings.bench_output_dir

    json_path = write_json_report(report, output_dir)
    md_path = write_markdown_report(report, output_dir, settings)

    return json_path, md_path
