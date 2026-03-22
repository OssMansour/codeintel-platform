"""
CodeIntel Benchmarks — Sample Test Data Generator

Produces a manifest.json with representative test cases that work
against any indexed CodeIntel instance. Cases are designed to be
adaptable: users can edit the manifest to match their specific repos.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.config import (
    AgentTestCase,
    BenchmarkManifest,
    RetrievalTestCase,
    WikiQualityTestCase,
    ensure_dir,
)


def _sample_retrieval_cases() -> list[RetrievalTestCase]:
    """Generate sample retrieval test cases."""
    return [
        RetrievalTestCase(
            query_id="ret-001",
            query="webhook handler for GitLab push events",
            expected_files=["gitlab_client.py", "main.py"],
            expected_symbols=["handle_push_event", "process_webhook"],
            expected_collection="code_repo",
            tags=["ingestion", "webhook"],
        ),
        RetrievalTestCase(
            query_id="ret-002",
            query="How are code embeddings generated and cached?",
            expected_files=["embedder.py"],
            expected_symbols=["embed_texts", "CodeEmbedder"],
            expected_collection="code_repo",
            tags=["indexing", "embedding"],
        ),
        RetrievalTestCase(
            query_id="ret-003",
            query="cross-encoder reranking implementation",
            expected_files=["reranker.py"],
            expected_symbols=["rerank", "CrossEncoderReranker"],
            expected_collection="code_repo",
            tags=["agent", "reranking"],
        ),
        RetrievalTestCase(
            query_id="ret-004",
            query="Qdrant hybrid search with BM25",
            expected_files=["qdrant_store.py"],
            expected_symbols=["hybrid_search", "QdrantStore"],
            expected_collection="code_repo",
            tags=["indexing", "search"],
        ),
        RetrievalTestCase(
            query_id="ret-005",
            query="tree-sitter parser for Python functions",
            expected_files=["parser.py"],
            expected_symbols=["parse_file", "TreeSitterParser"],
            expected_collection="code_repo",
            tags=["parsing"],
        ),
        RetrievalTestCase(
            query_id="ret-006",
            query="Celery task for wiki document generation",
            expected_files=["tasks.py"],
            expected_symbols=["generate_wiki", "generate_wiki_incremental"],
            expected_collection="code_repo",
            tags=["docgen", "celery"],
        ),
        RetrievalTestCase(
            query_id="ret-007",
            query="LLM provider fallback chain configuration",
            expected_files=["factory.py"],
            expected_symbols=["FallbackProvider", "create_fallback_provider"],
            expected_collection="code_repo",
            tags=["llm", "provider"],
        ),
        RetrievalTestCase(
            query_id="ret-008",
            query="dependency graph topological sort with cycle detection",
            expected_files=["dependency_graph.py"],
            expected_symbols=["topological_sort", "build_dependency_graph"],
            expected_collection="code_repo",
            tags=["parsing", "graph"],
        ),
        RetrievalTestCase(
            query_id="ret-009",
            query="hierarchical wiki module clustering algorithm",
            expected_files=["module_cluster.py", "wiki_generator.py"],
            expected_symbols=["cluster_components", "WikiGenerator"],
            expected_collection="code_repo",
            tags=["docgen", "clustering"],
        ),
        RetrievalTestCase(
            query_id="ret-010",
            query="iterative document refinement with critique scoring",
            expected_files=["doc_refiner.py"],
            expected_symbols=["DocRefiner", "refine"],
            expected_collection="code_repo",
            tags=["docgen", "refiner"],
        ),
    ]


def _sample_agent_cases() -> list[AgentTestCase]:
    """Generate sample agent regression test cases."""
    return [
        AgentTestCase(
            query_id="agent-001",
            query="How does the ingestion webhook handler verify HMAC signatures?",
            must_contain=["hmac", "signature"],
            expected_tool_calls=["search_code"],
            min_confidence=0.3,
            tags=["ingestion", "security"],
        ),
        AgentTestCase(
            query_id="agent-002",
            query="Explain how the cross-encoder reranker improves search results",
            must_contain=["rerank", "cross-encoder"],
            expected_tool_calls=["search_code"],
            min_confidence=0.3,
            tags=["agent", "reranking"],
        ),
        AgentTestCase(
            query_id="agent-003",
            query="What collections does Qdrant use and what is stored in each?",
            must_contain=["code_repo", "app_docs"],
            expected_tool_calls=["search_code"],
            min_confidence=0.3,
            tags=["indexing", "qdrant"],
        ),
        AgentTestCase(
            query_id="agent-004",
            query="How does the wiki generator handle module dependencies?",
            must_contain=["topological", "dependency"],
            expected_tool_calls=["search_code"],
            min_confidence=0.3,
            tags=["docgen"],
        ),
        AgentTestCase(
            query_id="agent-005",
            query="What programming languages does the tree-sitter parser support?",
            must_contain=["python", "javascript"],
            must_not_contain=["error", "unsupported"],
            expected_tool_calls=["search_code"],
            min_confidence=0.3,
            tags=["parsing"],
        ),
    ]


def _sample_wiki_quality_cases() -> list[WikiQualityTestCase]:
    """Generate sample wiki quality evaluation cases."""
    return [
        WikiQualityTestCase(
            project_id="codeintel",
            module_name="",  # Evaluate all modules
            min_overall=3.5,
        ),
    ]


def generate_sample_manifest(output_dir: str = "benchmarks/data") -> Path:
    """
    Generate a complete sample manifest.json.

    Returns the path to the written file.
    """
    manifest = BenchmarkManifest(
        retrieval_cases=_sample_retrieval_cases(),
        agent_cases=_sample_agent_cases(),
        wiki_quality_cases=_sample_wiki_quality_cases(),
    )

    out = ensure_dir(output_dir)
    filepath = out / "manifest.json"

    data = manifest.model_dump(mode="json")
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return filepath
