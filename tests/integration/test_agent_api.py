"""
Integration tests for services/agent/api.py (FastAPI agent service, port 8001).

Uses FastAPI's TestClient — no real network, no real models.
The lifespan context is replaced with a no-op so singletons are never
initialized; get_agent() and get_store() are patched directly.
"""
import sys
import os
import unittest.mock as mock
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent(fake_agent_response):
    agent = mock.MagicMock()
    agent.query.return_value = fake_agent_response
    return agent


@pytest.fixture
def mock_store_healthy():
    store = mock.MagicMock()
    store.get_all_collection_stats.return_value = {
        "code_repo": {"collection": "code_repo", "points_count": 100, "status": "green", "exists": True},
        "app_docs": {"collection": "app_docs", "points_count": 50, "status": "green", "exists": True},
        "incident_reports": {"collection": "incident_reports", "points_count": 10, "status": "green", "exists": True},
    }
    return store


@pytest.fixture
def client(mock_agent, mock_store_healthy):
    """FastAPI TestClient with all heavy singletons patched out."""

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    with (
        mock.patch("services.agent.agent.get_provider", return_value=mock.MagicMock()),
        mock.patch("services.agent.agent.get_reranker", return_value=mock.MagicMock()),
        mock.patch("services.agent.agent.os.makedirs"),
        mock.patch("services.agent.agent.sqlite3.connect", return_value=mock.MagicMock()),
    ):
        # Import app AFTER patching agent deps
        import importlib
        import services.agent.api as api_module
        importlib.reload(api_module)

        with (
            mock.patch.object(api_module, "get_agent", return_value=mock_agent),
            mock.patch.object(api_module, "get_store", return_value=mock_store_healthy),
        ):
            api_module.app.router.lifespan_context = noop_lifespan
            from fastapi.testclient import TestClient
            with TestClient(api_module.app, raise_server_exceptions=False) as tc:
                yield tc


# ===========================================================================
# /health
# ===========================================================================


def test_health_check_returns_healthy(client, mock_store_healthy):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "agent"
    assert body["qdrant_connected"] is True


def test_health_check_qdrant_false_on_exception(mock_agent):
    store = mock.MagicMock()
    store.get_all_collection_stats.side_effect = ConnectionError("qdrant down")

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    with (
        mock.patch("services.agent.agent.get_provider", return_value=mock.MagicMock()),
        mock.patch("services.agent.agent.get_reranker", return_value=mock.MagicMock()),
        mock.patch("services.agent.agent.os.makedirs"),
        mock.patch("services.agent.agent.sqlite3.connect", return_value=mock.MagicMock()),
    ):
        import importlib
        import services.agent.api as api_module
        importlib.reload(api_module)

        with (
            mock.patch.object(api_module, "get_agent", return_value=mock_agent),
            mock.patch.object(api_module, "get_store", return_value=store),
        ):
            api_module.app.router.lifespan_context = noop_lifespan
            from fastapi.testclient import TestClient
            with TestClient(api_module.app, raise_server_exceptions=False) as tc:
                resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json()["qdrant_connected"] is False


# ===========================================================================
# /agent/query
# ===========================================================================


def test_agent_query_200(client):
    resp = client.post("/agent/query", json={"query": "what is retry logic?", "project_id": "myproject"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "confidence" in body
    assert body["confidence"] == pytest.approx(0.85)


def test_agent_query_missing_field_422(client):
    resp = client.post("/agent/query", json={})
    assert resp.status_code == 422


def test_agent_query_empty_string_422(client):
    resp = client.post("/agent/query", json={"query": ""})
    assert resp.status_code == 422


def test_agent_query_500_on_exception(client, mock_agent):
    mock_agent.query.side_effect = RuntimeError("graph failed")
    resp = client.post("/agent/query", json={"query": "find retry logic"})
    assert resp.status_code == 500


# ===========================================================================
# /agent/localize
# ===========================================================================


def test_agent_localize_picks_code_repo_source(client, mock_agent, fake_agent_response):
    # Add a non-code source so we confirm the code_repo one wins
    from services.agent.agent import AgentResponse
    mock_agent.query.return_value = AgentResponse(
        answer="Found in auth.py line 42",
        sources=[
            {"file": "docs/overview.md", "symbol": "", "start_line": 0, "end_line": 0, "collection": "app_docs", "permalink": "", "score": 0.5, "section_path": ""},
            {"file": "auth.py", "symbol": "verify_token", "start_line": 42, "end_line": 65, "collection": "code_repo", "permalink": "", "score": 0.9, "section_path": ""},
        ],
        confidence=0.9,
    )
    resp = client.post("/agent/localize", json={"issue_text": "NullPointerException in auth module during token verification"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["file_path"] == "auth.py"
    assert body["localized"] is True


# ===========================================================================
# /agent/collections/status
# ===========================================================================


def test_collections_status_all_three(client):
    resp = client.get("/agent/collections/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "code_repo" in body
    assert "app_docs" in body
    assert "incident_reports" in body
