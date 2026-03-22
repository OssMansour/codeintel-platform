"""
CodeIntel Benchmarks — CLI Entry Point

Usage:
    python -m benchmarks.cli run --suite all
    python -m benchmarks.cli run --suite retrieval
    python -m benchmarks.cli run --suite wiki_quality
    python -m benchmarks.cli run --suite latency
    python -m benchmarks.cli run --suite regression
    python -m benchmarks.cli run --suite retrieval,latency
    python -m benchmarks.cli generate-sample-data

Environment variables (or .env):
    BENCH_AGENT_API_URL     Agent API base URL (default: http://localhost:8001)
    BENCH_QDRANT_URL        Qdrant URL (default: http://localhost:6333)
    BENCH_WIKI_DOCS_DIR     Wiki docs directory
    BENCH_OUTPUT_DIR        Output directory for reports
    BENCH_MIN_MRR           Minimum MRR threshold
    ...etc (see BenchmarkSettings)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import structlog

from benchmarks.config import (
    BenchmarkManifest,
    BenchmarkSettings,
    BenchmarkSuite,
    FullBenchmarkReport,
    now_iso,
    ensure_dir,
)
from benchmarks.report import generate_reports

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load test cases from a JSON manifest file."""
    p = Path(path)
    if not p.is_file():
        log.warning("cli.manifest_not_found", path=str(p))
        return BenchmarkManifest()

    data = json.loads(p.read_text(encoding="utf-8"))
    return BenchmarkManifest(**data)


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def run_suites(
    suites: list[str],
    manifest: BenchmarkManifest,
    settings: BenchmarkSettings,
) -> FullBenchmarkReport:
    """
    Run requested benchmark suites and return a combined report.
    """
    report = FullBenchmarkReport(
        timestamp=now_iso(),
        settings_snapshot={
            "agent_api_url": settings.agent_api_url,
            "qdrant_url": settings.qdrant_url,
            "wiki_docs_dir": settings.wiki_docs_dir,
            "min_mrr": settings.min_mrr,
            "min_hit_at_3": settings.min_hit_at_3,
            "min_hit_at_5": settings.min_hit_at_5,
            "min_wiki_quality": settings.min_wiki_quality,
            "max_agent_latency_s": settings.max_agent_latency_s,
        },
    )
    t_start = time.perf_counter()

    run_all = "all" in suites

    # --- Retrieval ---
    if run_all or "retrieval" in suites:
        if manifest.retrieval_cases:
            from benchmarks.retrieval import run_retrieval_benchmark

            log.info("cli.running_suite", suite="retrieval")
            report.retrieval = run_retrieval_benchmark(
                manifest.retrieval_cases, settings
            )
        else:
            log.warning("cli.no_cases", suite="retrieval")

    # --- Wiki Quality ---
    if run_all or "wiki_quality" in suites:
        if manifest.wiki_quality_cases:
            from benchmarks.wiki_quality import run_wiki_quality_benchmark

            log.info("cli.running_suite", suite="wiki_quality")
            report.wiki_quality = run_wiki_quality_benchmark(
                manifest.wiki_quality_cases, settings
            )
        else:
            log.warning("cli.no_cases", suite="wiki_quality")

    # --- Latency ---
    if run_all or "latency" in suites:
        from benchmarks.latency import run_latency_benchmark

        log.info("cli.running_suite", suite="latency")
        report.latency = run_latency_benchmark(settings)

    # --- Regression ---
    if run_all or "regression" in suites:
        if manifest.agent_cases:
            from benchmarks.regression import run_regression_tests

            log.info("cli.running_suite", suite="regression")
            report.regression = run_regression_tests(
                manifest.agent_cases, settings
            )
        else:
            log.warning("cli.no_cases", suite="regression")

    report.total_duration_s = time.perf_counter() - t_start
    return report


# ---------------------------------------------------------------------------
# Check pass/fail against thresholds
# ---------------------------------------------------------------------------


def check_thresholds(report: FullBenchmarkReport, settings: BenchmarkSettings) -> bool:
    """
    Check if the benchmark results meet configured thresholds.

    Returns True if all pass, False if any fail.
    """
    passed = True

    if report.retrieval:
        r = report.retrieval
        checks = [
            ("MRR", r.mean_mrr, settings.min_mrr),
            ("Hit@3", r.mean_hit_at_3, settings.min_hit_at_3),
            ("Hit@5", r.mean_hit_at_5, settings.min_hit_at_5),
            ("nDCG@5", r.mean_ndcg_at_5, settings.min_ndcg_at_5),
        ]
        for name, val, threshold in checks:
            if val < threshold:
                log.error("threshold.fail", metric=name, value=f"{val:.3f}", threshold=threshold)
                passed = False

    if report.wiki_quality:
        if report.wiki_quality.mean_overall < settings.min_wiki_quality:
            log.error(
                "threshold.fail",
                metric="wiki_quality",
                value=f"{report.wiki_quality.mean_overall:.2f}",
                threshold=settings.min_wiki_quality,
            )
            passed = False

    if report.regression:
        if report.regression.failed_cases > 0:
            log.error(
                "threshold.fail",
                metric="regression",
                failed=report.regression.failed_cases,
                total=report.regression.total_cases,
            )
            passed = False

    return passed


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Run benchmark suites."""
    settings = BenchmarkSettings()

    # Override from CLI args
    if args.output:
        settings.bench_output_dir = args.output
    if args.agent_url:
        settings.agent_api_url = args.agent_url

    # Load manifest
    manifest_path = args.manifest or Path(settings.bench_data_dir) / "manifest.json"
    manifest = load_manifest(manifest_path)

    # Parse suites
    suites = [s.strip() for s in args.suite.split(",")]

    # Run
    report = run_suites(suites, manifest, settings)

    # Generate reports
    json_path, md_path = generate_reports(report, settings)

    print(f"\n{'='*60}")
    print(f"  CodeIntel Benchmark Report")
    print(f"{'='*60}")

    if report.retrieval:
        r = report.retrieval
        print(f"\n  📊 Retrieval: MRR={r.mean_mrr:.3f}  Hit@5={r.mean_hit_at_5:.3f}"
              f"  nDCG@5={r.mean_ndcg_at_5:.3f}  ({r.passed_cases}/{r.total_cases} passed)")

    if report.wiki_quality:
        w = report.wiki_quality
        print(f"  📝 Wiki Quality: {w.mean_overall:.2f}/5.0"
              f"  ({w.passed_docs}/{w.total_docs} passed)")

    if report.latency:
        for name, p in report.latency.profiles.items():
            print(f"  ⏱️  {name}: mean={p.mean_ms:.0f}ms  p95={p.p95_ms:.0f}ms")

    if report.regression:
        reg = report.regression
        status = "PASS ✅" if reg.failed_cases == 0 else "FAIL ❌"
        print(f"  🧪 Regression: {reg.passed_cases}/{reg.total_cases} passed — {status}")

    print(f"\n  Duration: {report.total_duration_s:.1f}s")
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"{'='*60}\n")

    # Threshold check
    if args.check_thresholds:
        if not check_thresholds(report, settings):
            print("  ❌ THRESHOLD CHECK FAILED — see logs above")
            return 1
        print("  ✅ All thresholds passed")

    return 0


def cmd_generate_sample_data(args: argparse.Namespace) -> int:
    """Generate sample manifest with example test cases."""
    from benchmarks.sample_data import generate_sample_manifest

    output = args.output or "benchmarks/data"
    path = generate_sample_manifest(output)
    print(f"Sample manifest written to: {path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeintel-bench",
        description="CodeIntel Platform — Benchmarking & Quality Metrics",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run benchmark suites")
    run_parser.add_argument(
        "--suite", "-s",
        default="all",
        help="Comma-separated suites: retrieval,wiki_quality,latency,regression,all",
    )
    run_parser.add_argument(
        "--manifest", "-m",
        default=None,
        help="Path to test manifest JSON (default: benchmarks/data/manifest.json)",
    )
    run_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for reports",
    )
    run_parser.add_argument(
        "--agent-url",
        default=None,
        help="Agent API URL override",
    )
    run_parser.add_argument(
        "--check-thresholds",
        action="store_true",
        default=False,
        help="Exit with code 1 if thresholds are not met (for CI)",
    )

    # --- generate-sample-data ---
    sample_parser = subparsers.add_parser(
        "generate-sample-data",
        help="Generate a sample test manifest with example cases",
    )
    sample_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "generate-sample-data":
        return cmd_generate_sample_data(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
