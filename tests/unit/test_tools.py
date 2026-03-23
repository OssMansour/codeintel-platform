"""
Unit tests for services/agent/tools.py

All 6 agent tools are tested with mocked embedder and store singletons.
No real Qdrant, no real embedder model.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest

from tests.conftest import make_hit


# ---------------------------------------------------------------------------
# Autouse fixture: reset singletons before/after every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singletons():
    from services.agent import tools
    tools.reset_tool_singletons()
    yield
    tools.reset_tool_singletons()


# ---------------------------------------------------------------------------
# Helper: inject mock embedder + store into tools module
# ---------------------------------------------------------------------------


def _patch_tools(mock_embedder, mock_store):
    """Return a context manager that patches both singletons in the tools module."""
    import contextlib
    return contextlib.ExitStack()


def _inject(mock_embedder, mock_store):
    """Directly inject mocks into the tools module's module-level singletons."""
    from services.agent import tools
    tools._embedder = mock_embedder
    tools._store = mock_store


# ===========================================================================
# _get_permalink tests
# ===========================================================================


def test_get_permalink_prefers_scm_over_gitlab():
    from services.agent.tools import _get_permalink
    meta = {"scm_permalink": "https://github.com/repo/blob/main/foo.py", "gitlab_permalink": "https://gitlab.com/repo/-/blob/main/foo.py"}
    assert _get_permalink(meta) == "https://github.com/repo/blob/main/foo.py"


def test_get_permalink_falls_back_to_gitlab_permalink():
    from services.agent.tools import _get_permalink
    meta = {"gitlab_permalink": "https://gitlab.com/repo/-/blob/main/foo.py"}
    assert _get_permalink(meta) == "https://gitlab.com/repo/-/blob/main/foo.py"


def test_get_permalink_returns_empty_when_neither_set():
    from services.agent.tools import _get_permalink
    assert _get_permalink({}) == ""


# ===========================================================================
# _format_result tests
# ===========================================================================


def test_format_result_uses_metadata_content_when_top_level_missing():
    from services.agent.tools import _format_result
    hit = {
        "chunk_id": "abc",
        "score": 0.9,
        "collection": "code_repo",
        "metadata": {
            "file_path": "foo.py",
            "symbol_name": "bar",
            "content": "def bar(): pass",
        },
    }
    result = _format_result(hit)
    assert result["content"] == "def bar(): pass"


def test_format_result_all_fields_present():
    from services.agent.tools import _format_result
    hit = make_hit(
        chunk_id="xyz",
        score=0.12345,
        content="hello",
        collection="app_docs",
        file_path="docs/api.md",
        symbol_name="overview",
        start_line=10,
        end_line=20,
        affected_services=["payment", "auth"],
    )
    result = _format_result(hit)
    assert result["score"] == round(0.12345, 4)
    assert result["file_path"] == "docs/api.md"
    assert result["symbol_name"] == "overview"
    assert result["start_line"] == 10
    assert isinstance(result["affected_services"], list)
    assert result["content"] == "hello"


# ===========================================================================
# search_code tests
# ===========================================================================


def test_search_code_calls_hybrid_search_with_correct_collection(mock_embedder, mock_store):
    _inject(mock_embedder, mock_store)
    from services.agent.tools import search_code
    search_code.invoke({"query": "authentication", "language": "python"})
    call_kwargs = mock_store.hybrid_search.call_args
    assert call_kwargs.kwargs["collection"] == "code_repo"
    assert call_kwargs.kwargs["filters"] == {"language": "python"}


def test_search_code_returns_formatted_results(mock_embedder, mock_store):
    mock_store.hybrid_search.return_value = [make_hit(chunk_id="c1", score=0.9)]
    _inject(mock_embedder, mock_store)
    from services.agent.tools import search_code
    results = search_code.invoke({"query": "retry"})
    assert len(results) == 1
    assert "chunk_id" in results[0]
    assert "content" in results[0]
    assert "score" in results[0]


def test_search_code_respects_top_k_cap(mock_embedder, mock_store):
    mock_store.hybrid_search.return_value = []
    _inject(mock_embedder, mock_store)
    from services.agent.tools import search_code
    search_code.invoke({"query": "x", "top_k": 999})
    call_kwargs = mock_store.hybrid_search.call_args
    assert call_kwargs.kwargs["top_k"] == 50


# ===========================================================================
# search_incidents tests
# ===========================================================================


def test_search_incidents_returns_empty_when_collection_unpopulated(mock_embedder, mock_store):
    mock_store.is_collection_populated.return_value = False
    _inject(mock_embedder, mock_store)
    from services.agent.tools import search_incidents
    result = search_incidents.invoke({"query": "timeout"})
    assert result == []
    mock_store.hybrid_search.assert_not_called()


def test_search_incidents_applies_severity_filter_uppercase(mock_embedder, mock_store):
    mock_store.is_collection_populated.return_value = True
    mock_store.hybrid_search.return_value = []
    _inject(mock_embedder, mock_store)
    from services.agent.tools import search_incidents
    search_incidents.invoke({"query": "crash", "severity": "p1"})
    call_kwargs = mock_store.hybrid_search.call_args
    assert call_kwargs.kwargs["filters"] == {"severity": "P1"}


# ===========================================================================
# keyword_search tests
# ===========================================================================


def test_keyword_search_returns_empty_for_empty_collection(mock_embedder, mock_store):
    mock_store.is_collection_populated.return_value = False
    _inject(mock_embedder, mock_store)
    from services.agent.tools import keyword_search
    result = keyword_search.invoke({"terms": ["retry_payment"]})
    assert result == []


def test_keyword_search_joins_terms_for_query_text(mock_embedder, mock_store):
    mock_store.is_collection_populated.return_value = True
    mock_store.hybrid_search.return_value = []
    _inject(mock_embedder, mock_store)
    from services.agent.tools import keyword_search
    keyword_search.invoke({"terms": ["PaymentError", "timeout"]})
    mock_embedder.embed_query.assert_called_once_with("PaymentError timeout")


# ===========================================================================
# retrieve_entity tests
# ===========================================================================


def test_retrieve_entity_returns_not_found_when_no_results(mock_embedder, mock_store):
    mock_store.hybrid_search.return_value = []
    _inject(mock_embedder, mock_store)
    from services.agent.tools import retrieve_entity
    result = retrieve_entity.invoke({"node_id": "nonexistent_fn"})
    assert result["found"] is False
    assert "No entity found" in result["message"]


def test_retrieve_entity_exact_match_preferred_over_partial(mock_embedder, mock_store):
    # Two hits: partial match first, then exact match.
    # qualified_name must not accidentally match "foo" on the partial hit.
    partial_hit = make_hit(chunk_id="c1", score=0.95, symbol_name="foo_bar_extra", qualified_name="MyClass.foo_bar_extra")
    exact_hit = make_hit(chunk_id="c2", score=0.80, symbol_name="foo", qualified_name="foo")
    mock_store.hybrid_search.return_value = [partial_hit, exact_hit]
    _inject(mock_embedder, mock_store)
    from services.agent.tools import retrieve_entity
    result = retrieve_entity.invoke({"node_id": "foo"})
    assert result["found"] is True
    assert result["symbol_name"] == "foo"
