"""
CodeIntel Platform — Benchmarking & Quality Metrics (Phase 5)

Provides automated evaluation of retrieval quality, wiki generation quality,
end-to-end latency profiling, and a regression test harness.

Modules:
    config      — Benchmark configuration and shared data models
    retrieval   — MRR, Hit@K, nDCG retrieval benchmarks
    wiki_quality— Automated wiki doc scoring via critique prompts
    latency     — End-to-end latency profiler for all pipelines
    regression  — Regression test harness with assertion framework
    report      — JSON + Markdown report generator
    cli         — Click CLI entry point
"""
