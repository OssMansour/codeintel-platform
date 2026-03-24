"""
CodeIntel Benchmarks — Configuration & Shared Models

All benchmark settings, test case definitions, and result data models.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Settings (from environment / .env)
# ---------------------------------------------------------------------------


class BenchmarkSettings(BaseSettings):
    """Top-level benchmark settings loaded from environment."""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Agent API
    agent_api_url: str = "http://localhost:8001"

    # LLM (for wiki quality scoring)
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"

    # Paths
    bench_data_dir: str = "benchmarks/data"
    bench_output_dir: str = "benchmarks/results"
    repos_base_dir: str = "/data/repos"
    wiki_docs_dir: str = "/data/indexes/wiki"

    # Thresholds (for regression assertions)
    min_mrr: float = 0.40
    min_hit_at_3: float = 0.55
    min_hit_at_5: float = 0.70
    min_ndcg_at_5: float = 0.45
    min_wiki_quality: float = 3.5
    max_agent_latency_s: float = 30.0
    max_retrieval_latency_ms: float = 500.0

    model_config = ConfigDict(
        env_file=".env",
        env_prefix="BENCH_",
        case_sensitive=False,
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BenchmarkSuite(str, Enum):
    """Available benchmark suites."""

    RETRIEVAL = "retrieval"
    WIKI_QUALITY = "wiki_quality"
    LATENCY = "latency"
    REGRESSION = "regression"
    ALL = "all"


# ---------------------------------------------------------------------------
# Test Case Models (input)
# ---------------------------------------------------------------------------


class RetrievalTestCase(BaseModel):
    """A single retrieval quality test case."""

    query_id: str = Field(description="Unique test case ID")
    query: str = Field(description="Natural language search query")
    project_id: str = Field(default="", description="Project scope (empty = all)")
    expected_files: list[str] = Field(
        default_factory=list,
        description="File paths that MUST appear in results",
    )
    expected_symbols: list[str] = Field(
        default_factory=list,
        description="Symbol names that MUST appear in results",
    )
    expected_collection: str = Field(
        default="code_repo",
        description="Primary collection expected to contain the answer",
    )
    tags: list[str] = Field(default_factory=list, description="Category tags")


class AgentTestCase(BaseModel):
    """A single agent regression test case."""

    query_id: str
    query: str
    project_id: str = ""
    must_contain: list[str] = Field(
        default_factory=list,
        description="Substrings that MUST appear in the answer",
    )
    must_not_contain: list[str] = Field(
        default_factory=list,
        description="Substrings that MUST NOT appear in the answer",
    )
    min_confidence: float = Field(
        default=0.0,
        description="Minimum confidence threshold",
    )
    expected_tool_calls: list[str] = Field(
        default_factory=list,
        description="Tool names that should be invoked",
    )
    max_latency_s: float = Field(
        default=0.0,
        description="Override max latency for this case (0 = use global)",
    )
    tags: list[str] = Field(default_factory=list)


class WikiQualityTestCase(BaseModel):
    """A wiki quality evaluation target."""

    project_id: str
    module_name: str = Field(
        default="",
        description="Specific module to evaluate (empty = sample all)",
    )
    min_overall: float = Field(
        default=3.5,
        description="Minimum overall quality score",
    )


class BenchmarkManifest(BaseModel):
    """Full benchmark test manifest loaded from YAML/JSON."""

    retrieval_cases: list[RetrievalTestCase] = Field(default_factory=list)
    agent_cases: list[AgentTestCase] = Field(default_factory=list)
    wiki_quality_cases: list[WikiQualityTestCase] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result Models (output)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalCaseResult:
    """Result of a single retrieval test case."""

    query_id: str
    query: str
    reciprocal_rank: float = 0.0
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    latency_ms: float = 0.0
    result_count: int = 0
    matched_files: list[str] = field(default_factory=list)
    matched_symbols: list[str] = field(default_factory=list)
    top_results: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    failure_reason: str = ""


@dataclass
class RetrievalBenchmarkResult:
    """Aggregated retrieval benchmark results."""

    mean_mrr: float = 0.0
    mean_hit_at_1: float = 0.0
    mean_hit_at_3: float = 0.0
    mean_hit_at_5: float = 0.0
    mean_hit_at_10: float = 0.0
    mean_ndcg_at_5: float = 0.0
    mean_ndcg_at_10: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_cases: int = 0
    passed_cases: int = 0
    pass_rate: float = 0.0
    case_results: list[RetrievalCaseResult] = field(default_factory=list)
    timestamp: str = ""
    duration_s: float = 0.0


@dataclass
class WikiQualityCaseResult:
    """Result of evaluating a single wiki document."""

    project_id: str
    module_name: str
    scores: dict[str, int] = field(default_factory=dict)
    overall: float = 0.0
    issues: list[str] = field(default_factory=list)
    passed: bool = False
    doc_length_chars: int = 0
    latency_ms: float = 0.0


@dataclass
class WikiQualityResult:
    """Aggregated wiki quality benchmark results."""

    mean_overall: float = 0.0
    mean_completeness: float = 0.0
    mean_accuracy: float = 0.0
    mean_diagrams: float = 0.0
    mean_clarity: float = 0.0
    mean_cross_refs: float = 0.0
    total_docs: int = 0
    passed_docs: int = 0
    pass_rate: float = 0.0
    case_results: list[WikiQualityCaseResult] = field(default_factory=list)
    timestamp: str = ""
    duration_s: float = 0.0


@dataclass
class LatencyProfile:
    """Latency measurements for a single operation."""

    operation: str
    samples: list[float] = field(default_factory=list)  # ms

    @property
    def mean_ms(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    @property
    def min_ms(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def _percentile(self, pct: int) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(len(s) * pct / 100)
        return s[min(idx, len(s) - 1)]


@dataclass
class LatencyBenchmarkResult:
    """Aggregated latency profiling results."""

    profiles: dict[str, LatencyProfile] = field(default_factory=dict)
    timestamp: str = ""
    duration_s: float = 0.0


@dataclass
class AgentCaseResult:
    """Result of a single agent regression test."""

    query_id: str
    query: str
    answer_snippet: str = ""
    confidence: float = 0.0
    tool_calls_made: list[str] = field(default_factory=list)
    collection_hits: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    passed: bool = False
    failures: list[str] = field(default_factory=list)


@dataclass
class RegressionResult:
    """Aggregated regression test results."""

    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0
    agent_results: list[AgentCaseResult] = field(default_factory=list)
    timestamp: str = ""
    duration_s: float = 0.0


@dataclass
class FullBenchmarkReport:
    """Top-level container holding all benchmark results."""

    retrieval: RetrievalBenchmarkResult | None = None
    wiki_quality: WikiQualityResult | None = None
    latency: LatencyBenchmarkResult | None = None
    regression: RegressionResult | None = None
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    total_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """ISO-8601 timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
