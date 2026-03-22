"""
CodeIntel Platform — Ingestion Service
FastAPI application for receiving SCM webhooks (GitLab & GitHub) and
triggering indexing tasks.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Callable

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from services.docgen.tasks import index_changed_files, index_mr_context, index_repository, index_static_docs
from services.ingestion.scm_provider import SCMProvider

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class IngestionSettings(BaseSettings):
    """Configuration for the ingestion service loaded from environment."""

    # GitLab settings
    gitlab_webhook_secret: str = Field("", env="GITLAB_WEBHOOK_SECRET")
    gitlab_project_id: str = Field("", env="GITLAB_PROJECT_ID")

    # GitHub settings
    github_webhook_secret: str = Field("", env="GITHUB_WEBHOOK_SECRET")
    github_repo: str = Field("", env="GITHUB_REPO")

    # Common
    scm_provider: SCMProvider = Field(SCMProvider.GITLAB, env="SCM_PROVIDER")
    repo_clone_base: str = Field("/data/repos", env="REPO_CLONE_BASE")
    log_level: str = Field("info", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = IngestionSettings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), settings.log_level.upper(), 20)
    ),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

APP_VERSION = "1.1.0"

app = FastAPI(
    title="CodeIntel Ingestion Service",
    description="Receives GitLab / GitHub webhooks and dispatches indexing tasks.",
    version=APP_VERSION,
)

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class FullIndexRequest(BaseModel):
    """Request body for triggering a full repository index."""

    project_id: str = Field(..., description="SCM project identifier (GitLab namespace/path or GitHub owner/repo)")
    repo_path: str = Field(..., description="Local path to the cloned repository")
    commit_sha: str = Field("HEAD", description="Commit SHA to index")


class FileIndexRequest(BaseModel):
    """Request body for triggering a single file re-index."""

    project_id: str = Field(..., description="SCM project identifier (GitLab namespace/path or GitHub owner/repo)")
    file_path: str = Field(..., description="Relative file path within the repository")
    commit_sha: str = Field("HEAD", description="Commit SHA for this file version")
    repo_path: str = Field(..., description="Local path to the cloned repository")


class WebhookResponse(BaseModel):
    """Standard webhook acknowledgment response."""

    status: str
    message: str
    task_id: str | None = None


# ---------------------------------------------------------------------------
# HMAC Verification — GitLab
# ---------------------------------------------------------------------------


def _verify_gitlab_signature(
    body: bytes,
    signature: str | None,
    secret: str,
) -> bool:
    """
    Verify GitLab webhook HMAC-SHA256 signature.

    GitLab sends the signature in the X-Gitlab-Token header.
    For token-based auth, GitLab sends the raw secret, not HMAC.
    We support both modes: simple token comparison and HMAC.
    """
    if signature is None:
        return False

    # GitLab simple token verification (most common)
    if signature == secret:
        return True

    # HMAC-SHA256 verification (for advanced webhook configurations)
    # GitLab sends the raw hex digest in X-Gitlab-Token — no "sha256=" prefix.
    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    # Strip "sha256=" prefix from incoming signature if present (defensive)
    clean_sig = signature.removeprefix("sha256=")
    return hmac.compare_digest(clean_sig, expected)


async def _verify_and_parse_body(
    request: Request,
    signature: str | None,
    secret: str,
    verify_fn: Callable[[bytes, str | None, str], bool],
    provider_label: str,
) -> tuple[bytes, dict[str, Any]]:
    """
    Shared webhook body verification and JSON parsing.

    Reads the raw body, verifies the cryptographic signature using the
    supplied *verify_fn*, and parses the JSON payload.

    Raises:
        HTTPException 401: Signature mismatch.
        HTTPException 400: Invalid JSON payload.
    """
    body = await request.body()

    if not verify_fn(body, signature, secret):
        log.warning(
            "webhook_signature_verification_failed",
            provider=provider_label,
            remote=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {provider_label} webhook signature",
        )

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        log.error("webhook_invalid_json", provider=provider_label, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        ) from exc

    return body, payload


async def get_verified_body(
    request: Request,
    x_gitlab_token: str | None = Header(None, alias="X-Gitlab-Token"),
) -> tuple[bytes, dict[str, Any]]:
    """FastAPI dependency: verify GitLab webhook signature and parse body."""
    return await _verify_and_parse_body(
        request, x_gitlab_token, settings.gitlab_webhook_secret,
        _verify_gitlab_signature, "GitLab",
    )


# ---------------------------------------------------------------------------
# HMAC Verification — GitHub
# ---------------------------------------------------------------------------


def _verify_github_signature(
    body: bytes,
    signature: str | None,
    secret: str,
) -> bool:
    """
    Verify GitHub webhook HMAC-SHA256 signature.

    GitHub sends ``X-Hub-Signature-256: sha256=<hex>`` on every webhook
    delivery when a secret is configured.
    """
    if not signature:
        return False

    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature, f"sha256={expected}")


async def get_verified_github_body(
    request: Request,
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
) -> tuple[bytes, dict[str, Any]]:
    """FastAPI dependency: verify GitHub webhook signature and parse body."""
    return await _verify_and_parse_body(
        request, x_hub_signature_256, settings.github_webhook_secret,
        _verify_github_signature, "GitHub",
    )


# ---------------------------------------------------------------------------
# Helper: Extract Changed Files from Push Event
# ---------------------------------------------------------------------------


def _extract_changed_files(payload: dict[str, Any]) -> list[str]:
    """
    Extract the list of changed file paths from a push event payload.

    Works with both GitLab and GitHub push event formats:
    - GitLab: ``payload["commits"][*]["added"|"modified"|"removed"]``
    - GitHub: ``payload["commits"][*]["added"|"modified"|"removed"]``
    (Fortunately both platforms use the same structure for push commits.)
    """
    changed: set[str] = set()
    for commit in payload.get("commits", []):
        changed.update(commit.get("added", []))
        changed.update(commit.get("modified", []))
        changed.update(commit.get("removed", []))
    return list(changed)


def _extract_removed_files(payload: dict[str, Any]) -> list[str]:
    """Extract only removed files from a push event payload (GitLab & GitHub)."""
    removed: set[str] = set()
    for commit in payload.get("commits", []):
        removed.update(commit.get("removed", []))
    return list(removed)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Infrastructure"])
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns service status and version for Docker health checks and
    monitoring systems.
    """
    return {"status": "healthy", "service": "ingestion", "version": APP_VERSION}


@app.post(
    "/webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Webhooks"],
)
async def gitlab_webhook(
    x_gitlab_event: str | None = Header(None, alias="X-Gitlab-Event"),
    verified: tuple[bytes, dict[str, Any]] = Depends(get_verified_body),
) -> WebhookResponse:
    """
    Receive and process GitLab webhook events.

    Supported event types:
    - Push Hook: Triggers incremental re-indexing of changed files
    - Merge Request Hook: Logs MR metadata for context
    - Issue Hook: Logs issue metadata for context

    All processing is asynchronous. This endpoint always returns 202
    within 200ms after dispatching the Celery task.
    """
    _, payload = verified
    event_type = x_gitlab_event or payload.get("object_kind", "unknown")

    log.info(
        "webhook_received",
        provider="gitlab",
        event_type=event_type,
        project=payload.get("project", {}).get("path_with_namespace", "unknown"),
    )

    if event_type in ("Push Hook", "push"):
        return await _handle_gitlab_push_event(payload)
    elif event_type in ("Merge Request Hook", "merge_request"):
        return await _handle_gitlab_mr_event(payload)
    elif event_type in ("Issue Hook", "issue"):
        return await _handle_gitlab_issue_event(payload)
    else:
        log.info("webhook_event_ignored", event_type=event_type)
        return WebhookResponse(
            status="ignored",
            message=f"Event type '{event_type}' is not processed",
        )


# ---------------------------------------------------------------------------
# GitHub Webhook Endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/webhook/github",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Webhooks"],
)
async def github_webhook(
    x_github_event: str | None = Header(None, alias="X-GitHub-Event"),
    verified: tuple[bytes, dict[str, Any]] = Depends(get_verified_github_body),
) -> WebhookResponse:
    """
    Receive and process GitHub webhook events.

    Supported event types:
    - push: Triggers incremental re-indexing of changed files
    - pull_request: Logs PR metadata for context
    - issues: Logs issue metadata for context

    All processing is asynchronous. This endpoint always returns 202
    within 200ms after dispatching the Celery task.
    """
    _, payload = verified
    event_type = x_github_event or "unknown"

    log.info(
        "webhook_received",
        provider="github",
        event_type=event_type,
        repo=payload.get("repository", {}).get("full_name", "unknown"),
    )

    if event_type == "push":
        return await _handle_github_push_event(payload)
    elif event_type == "pull_request":
        return await _handle_github_pr_event(payload)
    elif event_type == "issues":
        return await _handle_github_issue_event(payload)
    elif event_type == "ping":
        return WebhookResponse(
            status="ok",
            message="GitHub webhook ping received",
        )
    else:
        log.info("github_webhook_event_ignored", event_type=event_type)
        return WebhookResponse(
            status="ignored",
            message=f"GitHub event type '{event_type}' is not processed",
        )


async def _dispatch_push_indexing(
    project_id: str,
    payload: dict[str, Any],
    provider_label: str,
) -> WebhookResponse:
    """
    Shared push-event handler for any SCM provider.

    Extracts changed/removed files from the payload, then dispatches
    an ``index_changed_files`` Celery task.
    """
    commit_sha = payload.get("after", "HEAD")
    ref = payload.get("ref", "refs/heads/main")
    branch = ref.replace("refs/heads/", "")

    changed_files = _extract_changed_files(payload)
    removed_files = _extract_removed_files(payload)

    if not changed_files and not removed_files:
        log.info(
            "push_event_no_changes",
            provider=provider_label,
            project_id=project_id,
            commit_sha=commit_sha,
        )
        return WebhookResponse(
            status="no_changes",
            message="Push event contained no file changes",
        )

    log.info(
        "push_event_dispatching",
        provider=provider_label,
        project_id=project_id,
        commit_sha=commit_sha,
        branch=branch,
        changed_count=len(changed_files),
        removed_count=len(removed_files),
    )

    repo_path = os.path.join(settings.repo_clone_base, project_id.replace("/", "_"))

    task = index_changed_files.apply_async(
        kwargs={
            "project_id": project_id,
            "changed_files": changed_files,
            "removed_files": removed_files,
            "commit_sha": commit_sha,
            "repo_path": repo_path,
            "branch": branch,
        },
        queue="indexing",
    )

    return WebhookResponse(
        status="accepted",
        message=f"Dispatched indexing for {len(changed_files)} changed files",
        task_id=task.id,
    )


async def _handle_gitlab_push_event(payload: dict[str, Any]) -> WebhookResponse:
    """Handle GitLab push events by triggering incremental file indexing."""
    project_id = str(
        payload.get("project", {}).get("path_with_namespace")
        or payload.get("project_id", settings.gitlab_project_id)
    )
    return await _dispatch_push_indexing(project_id, payload, "gitlab")


async def _handle_gitlab_mr_event(payload: dict[str, Any]) -> WebhookResponse:
    """
    Handle GitLab Merge Request events.

    Indexes MR descriptions and comments into app_docs for agent context.
    """
    mr = payload.get("object_attributes", {})
    project_id = str(
        payload.get("project", {}).get("path_with_namespace")
        or payload.get("project_id", settings.gitlab_project_id)
    )
    log.info(
        "gitlab_mr_event_received",
        mr_id=mr.get("iid"),
        title=mr.get("title"),
        state=mr.get("state"),
        action=mr.get("action"),
    )

    task = index_mr_context.apply_async(
        kwargs={
            "project_id": project_id,
            "context_type": "merge_request",
            "context_id": mr.get("iid", 0),
            "title": mr.get("title", ""),
            "body": mr.get("description", ""),
            "state": mr.get("state", ""),
            "author": mr.get("author", {}).get("username", "") if isinstance(mr.get("author"), dict) else str(mr.get("author_id", "")),
            "url": mr.get("url", ""),
        },
        queue="indexing",
    )

    return WebhookResponse(
        status="accepted",
        message=f"MR #{mr.get('iid')} dispatched for indexing",
        task_id=task.id,
    )


async def _handle_gitlab_issue_event(payload: dict[str, Any]) -> WebhookResponse:
    """
    Handle GitLab Issue events.

    Indexes issue descriptions into app_docs for agent context.
    """
    issue = payload.get("object_attributes", {})
    project_id = str(
        payload.get("project", {}).get("path_with_namespace")
        or payload.get("project_id", settings.gitlab_project_id)
    )
    log.info(
        "gitlab_issue_event_received",
        issue_id=issue.get("iid"),
        title=issue.get("title"),
        state=issue.get("state"),
        action=issue.get("action"),
    )

    task = index_mr_context.apply_async(
        kwargs={
            "project_id": project_id,
            "context_type": "issue",
            "context_id": issue.get("iid", 0),
            "title": issue.get("title", ""),
            "body": issue.get("description", ""),
            "state": issue.get("state", ""),
            "author": issue.get("author", {}).get("username", "") if isinstance(issue.get("author"), dict) else str(issue.get("author_id", "")),
            "url": issue.get("url", ""),
        },
        queue="indexing",
    )

    return WebhookResponse(
        status="accepted",
        message=f"Issue #{issue.get('iid')} dispatched for indexing",
        task_id=task.id,
    )


# ---------------------------------------------------------------------------
# GitHub Event Handlers
# ---------------------------------------------------------------------------


async def _handle_github_push_event(payload: dict[str, Any]) -> WebhookResponse:
    """Handle GitHub push events by triggering incremental file indexing."""
    repo_full_name = payload.get("repository", {}).get("full_name", settings.github_repo)
    return await _dispatch_push_indexing(repo_full_name, payload, "github")


async def _handle_github_pr_event(payload: dict[str, Any]) -> WebhookResponse:
    """
    Handle GitHub Pull Request events.

    Indexes PR descriptions and review comments into app_docs for agent context.
    """
    pr = payload.get("pull_request", {})
    action = payload.get("action", "unknown")
    repo_full_name = payload.get("repository", {}).get("full_name", settings.github_repo)
    log.info(
        "github_pr_event_received",
        pr_number=pr.get("number"),
        title=pr.get("title"),
        state=pr.get("state"),
        action=action,
    )

    task = index_mr_context.apply_async(
        kwargs={
            "project_id": repo_full_name,
            "context_type": "pull_request",
            "context_id": pr.get("number", 0),
            "title": pr.get("title", ""),
            "body": pr.get("body", "") or "",
            "state": pr.get("state", ""),
            "author": pr.get("user", {}).get("login", ""),
            "url": pr.get("html_url", ""),
        },
        queue="indexing",
    )

    return WebhookResponse(
        status="accepted",
        message=f"PR #{pr.get('number')} dispatched for indexing ({action})",
        task_id=task.id,
    )


async def _handle_github_issue_event(payload: dict[str, Any]) -> WebhookResponse:
    """
    Handle GitHub Issue events.

    Indexes issue descriptions into app_docs for agent context.
    """
    issue = payload.get("issue", {})
    action = payload.get("action", "unknown")
    repo_full_name = payload.get("repository", {}).get("full_name", settings.github_repo)
    log.info(
        "github_issue_event_received",
        issue_number=issue.get("number"),
        title=issue.get("title"),
        state=issue.get("state"),
        action=action,
    )

    task = index_mr_context.apply_async(
        kwargs={
            "project_id": repo_full_name,
            "context_type": "issue",
            "context_id": issue.get("number", 0),
            "title": issue.get("title", ""),
            "body": issue.get("body", "") or "",
            "state": issue.get("state", ""),
            "author": issue.get("user", {}).get("login", ""),
            "url": issue.get("html_url", ""),
        },
        queue="indexing",
    )

    return WebhookResponse(
        status="accepted",
        message=f"Issue #{issue.get('number')} dispatched for indexing ({action})",
        task_id=task.id,
    )


@app.post(
    "/index/full",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Indexing"],
)
async def trigger_full_index(request: FullIndexRequest) -> WebhookResponse:
    """
    Trigger a full repository re-index.

    This will re-parse and re-embed every file in the repository.
    Use this after initial setup or after major refactors.
    Processing is asynchronous.
    """
    log.info(
        "full_index_triggered",
        project_id=request.project_id,
        repo_path=request.repo_path,
        commit_sha=request.commit_sha,
    )

    task = index_repository.apply_async(
        kwargs={
            "project_id": request.project_id,
            "repo_path": request.repo_path,
            "commit_sha": request.commit_sha,
        },
        queue="indexing",
    )

    return WebhookResponse(
        status="accepted",
        message=f"Full index dispatched for project {request.project_id}",
        task_id=task.id,
    )


@app.post(
    "/index/file",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Indexing"],
)
async def trigger_file_index(request: FileIndexRequest) -> WebhookResponse:
    """
    Trigger re-indexing of a single file.

    Useful for manual re-indexing of a specific file without triggering
    a full repository index.
    """
    log.info(
        "file_index_triggered",
        project_id=request.project_id,
        file_path=request.file_path,
        commit_sha=request.commit_sha,
    )

    task = index_changed_files.apply_async(
        kwargs={
            "project_id": request.project_id,
            "changed_files": [request.file_path],
            "removed_files": [],
            "commit_sha": request.commit_sha,
            "repo_path": request.repo_path,
            "branch": "main",
        },
        queue="indexing",
    )

    return WebhookResponse(
        status="accepted",
        message=f"File index dispatched for {request.file_path}",
        task_id=task.id,
    )


@app.post(
    "/index/static-docs",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Indexing"],
)
async def trigger_static_docs_index() -> WebhookResponse:
    """
    Trigger re-indexing of Application_documentation.md and incident_reports.md.

    Call this after updating either static document file.
    """
    log.info("static_docs_index_triggered")

    task = index_static_docs.apply_async(queue="indexing")

    return WebhookResponse(
        status="accepted",
        message="Static document re-indexing dispatched",
        task_id=task.id,
    )


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions to ensure structured logging."""
    log.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
