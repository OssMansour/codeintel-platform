"""
Integration tests for services/ingestion/main.py (FastAPI ingestion service, port 8000).

Uses FastAPI TestClient — no real Celery, no real SCM API.
Celery tasks are patched before the app module is reloaded so they
never try to connect to Redis.

HMAC helpers:
  - GitLab uses a simple secret token comparison
  - GitHub uses HMAC-SHA256 (sha256=<hex>)
"""
import hashlib
import hmac
import importlib
import json
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------


def _gitlab_token_header(secret: str) -> dict:
    """GitLab uses plain secret token in X-Gitlab-Token header."""
    return {"X-Gitlab-Token": secret}


def _github_sig_header(body: bytes, secret: str) -> dict:
    """GitHub uses HMAC-SHA256 in X-Hub-Signature-256 header."""
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Hub-Signature-256": sig}


# ---------------------------------------------------------------------------
# Fixture: ingestion TestClient with celery tasks mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def ingestion_client(monkeypatch):
    monkeypatch.setenv("GITLAB_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "gh-secret")
    monkeypatch.setenv("GITLAB_PROJECT_ID", "mygroup/myrepo")

    mock_icf = mock.MagicMock()
    mock_icf.apply_async.return_value = mock.MagicMock(id="task-icf")
    mock_imc = mock.MagicMock()
    mock_imc.apply_async.return_value = mock.MagicMock(id="task-imc")
    mock_ir = mock.MagicMock()
    mock_ir.apply_async.return_value = mock.MagicMock(id="task-ir")
    mock_isd = mock.MagicMock()
    mock_isd.apply_async.return_value = mock.MagicMock(id="task-isd")

    with (
        mock.patch("services.docgen.tasks.index_changed_files", mock_icf),
        mock.patch("services.docgen.tasks.index_mr_context", mock_imc),
        mock.patch("services.docgen.tasks.index_repository", mock_ir),
        mock.patch("services.docgen.tasks.index_static_docs", mock_isd),
    ):
        import services.ingestion.main as ingestion_main
        importlib.reload(ingestion_main)
        from fastapi.testclient import TestClient
        with TestClient(ingestion_main.app, raise_server_exceptions=False) as tc:
            yield tc, mock_icf, mock_imc, ingestion_main


# ===========================================================================
# /health
# ===========================================================================


def test_health_check_returns_ingestion_service(ingestion_client):
    client, _, _, _ = ingestion_client
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "ingestion"


# ===========================================================================
# /webhook (GitLab)
# ===========================================================================


def test_gitlab_push_dispatches_index_task(ingestion_client):
    client, mock_icf, _, _ = ingestion_client
    payload = {
        "object_kind": "push",
        "project": {"id": "mygroup/myrepo", "path_with_namespace": "mygroup/myrepo"},
        "commits": [
            {"added": [], "modified": ["src/app.py"], "removed": []}
        ],
        "ref": "refs/heads/main",
        "checkout_sha": "abc123",
    }
    body = json.dumps(payload).encode()
    resp = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Gitlab-Event": "Push Hook", **_gitlab_token_header("test-secret")},
    )
    assert resp.status_code == 202
    body_json = resp.json()
    assert body_json["status"] == "accepted"
    mock_icf.apply_async.assert_called_once()


def test_gitlab_invalid_signature_returns_401(ingestion_client):
    client, _, _, _ = ingestion_client
    payload = {"object_kind": "push", "commits": []}
    body = json.dumps(payload).encode()
    resp = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Gitlab-Event": "Push Hook", "X-Gitlab-Token": "wrong-token"},
    )
    assert resp.status_code == 401


def test_gitlab_push_no_changes_returns_no_changes_status(ingestion_client):
    client, mock_icf, _, _ = ingestion_client
    payload = {
        "object_kind": "push",
        "project": {"id": "mygroup/myrepo", "path_with_namespace": "mygroup/myrepo"},
        "commits": [],
        "ref": "refs/heads/main",
        "checkout_sha": "abc123",
    }
    body = json.dumps(payload).encode()
    resp = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Gitlab-Event": "Push Hook", **_gitlab_token_header("test-secret")},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "no_changes"
    mock_icf.apply_async.assert_not_called()


# ===========================================================================
# /webhook/github (GitHub)
# ===========================================================================


def test_github_push_dispatches_index_task(ingestion_client):
    client, mock_icf, _, _ = ingestion_client
    payload = {
        "ref": "refs/heads/main",
        "repository": {"full_name": "myorg/myrepo"},
        "head_commit": {"id": "abc123"},
        "commits": [
            {"added": [], "modified": ["lib/payment.js"], "removed": []}
        ],
    }
    body = json.dumps(payload).encode()
    resp = client.post(
        "/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "push",
            **_github_sig_header(body, "gh-secret"),
        },
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


def test_github_ping_returns_ok(ingestion_client):
    client, _, _, _ = ingestion_client
    payload = {"zen": "Practicality beats purity.", "hook_id": 1}
    body = json.dumps(payload).encode()
    resp = client.post(
        "/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "ping",
            **_github_sig_header(body, "gh-secret"),
        },
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "ok"


# ===========================================================================
# _extract_changed_files — pure function, no mock needed
# ===========================================================================


def test_extract_changed_files_deduplicates_across_commits(ingestion_client):
    _, _, _, ingestion_main = ingestion_client
    payload = {
        "commits": [
            {"added": [], "modified": ["a.py"], "removed": []},
            {"added": ["a.py", "b.py"], "modified": [], "removed": []},
        ]
    }
    result = ingestion_main._extract_changed_files(payload)
    # Should deduplicate: a.py appears twice, b.py once
    assert set(result) == {"a.py", "b.py"}
    # No duplicates
    assert len(result) == len(set(result))
