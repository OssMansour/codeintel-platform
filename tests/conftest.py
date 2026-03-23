"""
Shared pytest fixtures for the CodeIntel Platform test suite.

Stubs out celery before any service import so that tests don't require
a running Redis broker. This mirrors the pattern used in test_fixes.py.
"""
import sys
import unittest.mock as mock

# ---------------------------------------------------------------------------
# Module-level stubs — must happen before any service import
# ---------------------------------------------------------------------------

sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_hit(
    chunk_id: str = "chunk_001",
    score: float = 0.85,
    content: str = "def foo(): pass",
    collection: str = "code_repo",
    **meta,
) -> dict:
    """
    Build a valid raw Qdrant search result dict.

    The shape mirrors what QdrantStore.hybrid_search() returns:
    top-level keys: chunk_id, score, content, metadata, collection.
    """
    metadata = {
        "file_path": meta.get("file_path", "services/foo.py"),
        "symbol_name": meta.get("symbol_name", "foo"),
        "symbol_type": meta.get("symbol_type", "function"),
        "qualified_name": meta.get("qualified_name", "foo"),
        "parent_symbol": meta.get("parent_symbol", ""),
        "start_line": meta.get("start_line", 1),
        "end_line": meta.get("end_line", 5),
        "language": meta.get("language", "python"),
        "project_id": meta.get("project_id", "myproject"),
        "branch": meta.get("branch", "main"),
        "commit_sha": meta.get("commit_sha", "abc123"),
        "scm_permalink": meta.get("scm_permalink", ""),
        "gitlab_permalink": meta.get("gitlab_permalink", ""),
        "source_collection": meta.get("source_collection", collection),
        "trust_level": meta.get("trust_level", "code"),
        "section_path": meta.get("section_path", ""),
        "heading": meta.get("heading", ""),
        "source_label": meta.get("source_label", ""),
        "severity": meta.get("severity", ""),
        "incident_date": meta.get("incident_date", ""),
        "affected_services": meta.get("affected_services", []),
        "calls": meta.get("calls", []),
        "params": meta.get("params", []),
        "content": content,
    }
    return {
        "chunk_id": chunk_id,
        "score": score,
        "content": content,
        "metadata": metadata,
        "collection": collection,
    }


# ---------------------------------------------------------------------------
# Embedder fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder():
    """Return a MagicMock that mimics CodeRankEmbedder."""
    embedder = mock.MagicMock()
    embedder.embed_query.return_value = [0.1] * 768
    embedder.embed_texts.return_value = [[0.1] * 768]
    embedder.dimension = 768
    return embedder


# ---------------------------------------------------------------------------
# Store fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    """Return a MagicMock that mimics QdrantStore."""
    store = mock.MagicMock()
    store.hybrid_search.return_value = []
    store.is_collection_populated.return_value = True
    store.get_all_collection_stats.return_value = {
        "code_repo": {"collection": "code_repo", "points_count": 100, "status": "green", "exists": True},
        "app_docs": {"collection": "app_docs", "points_count": 50, "status": "green", "exists": True},
        "incident_reports": {"collection": "incident_reports", "points_count": 10, "status": "green", "exists": True},
    }
    return store


# ---------------------------------------------------------------------------
# Fake AgentResponse fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_agent_response():
    """Return a canned AgentResponse for API integration tests."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Stub celery before importing agent module
    from services.agent.agent import AgentResponse
    return AgentResponse(
        answer="The retry logic is in services/payment/retry.py at line 42.",
        sources=[
            {
                "file": "services/payment/retry.py",
                "symbol": "retry_payment",
                "start_line": 42,
                "end_line": 65,
                "collection": "code_repo",
                "permalink": "https://gitlab.example.com/repo/-/blob/main/services/payment/retry.py#L42",
                "score": 0.92,
                "section_path": "",
            }
        ],
        confidence=0.85,
        collection_hits={"code_repo": 3, "app_docs": 1, "incident_reports": 0},
        graph_path=[],
        tool_calls_made=["search_code", "retrieve_entity"],
    )
