"""
CodeIntel Platform — SCM Provider Abstraction
Unified interface for interacting with Git hosting platforms (GitLab, GitHub).

This module defines the abstract base class that both GitLabClient and
GitHubClient implement, allowing the rest of the pipeline to be
SCM-agnostic.
"""

from __future__ import annotations

import enum
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# SCM Provider Enum
# ---------------------------------------------------------------------------


class SCMProvider(str, enum.Enum):
    """Supported source-code management platforms."""

    GITLAB = "gitlab"
    GITHUB = "github"


# ---------------------------------------------------------------------------
# Shared Data Classes (platform-neutral)
# ---------------------------------------------------------------------------


@dataclass
class DiffFile:
    """Represents a single changed file in a git diff."""

    old_path: str
    new_path: str
    new_file: bool
    renamed_file: bool
    deleted_file: bool
    diff: str
    a_mode: str = ""
    b_mode: str = ""


@dataclass
class CommitInfo:
    """Metadata about a single commit."""

    sha: str
    message: str
    author_name: str
    author_email: str
    authored_date: str
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


@dataclass
class PRInfo:
    """Metadata about a Pull / Merge Request (unified)."""

    id: int
    number: int  # iid for GitLab, number for GitHub
    title: str
    description: str
    state: str
    author: str
    source_branch: str
    target_branch: str
    created_at: str
    updated_at: str
    web_url: str


@dataclass
class IssueInfo:
    """Metadata about an Issue."""

    id: int
    number: int  # iid for GitLab, number for GitHub
    title: str
    description: str
    state: str
    author: str
    labels: list[str]
    created_at: str
    updated_at: str
    web_url: str


@dataclass
class CommentInfo:
    """A comment / note on a PR or Issue."""

    id: int
    author: str
    body: str
    created_at: str
    updated_at: str
    system: bool = False


# ---------------------------------------------------------------------------
# Abstract SCM Client
# ---------------------------------------------------------------------------


class SCMClient(ABC):
    """
    Abstract base class for SCM platform clients.

    Both ``GitLabClient`` and ``GitHubClient`` implement this interface so
    that the ingestion pipeline, chunker, and webhook handler can work
    with either platform transparently.
    """

    @property
    @abstractmethod
    def provider(self) -> SCMProvider:
        """Return the SCM provider enum value."""

    @property
    @abstractmethod
    def base_web_url(self) -> str:
        """Return the base web URL for permalink generation (e.g. repo web URL)."""

    # ---- Repository operations -------------------------------------------

    @abstractmethod
    def clone_or_pull(
        self,
        project_id: str | None = None,
        local_path: str | None = None,
        branch: str = "main",
    ) -> str:
        """Clone (or pull latest changes for) the repository. Returns local path."""

    # ---- File content ----------------------------------------------------

    @abstractmethod
    def get_file_content(
        self,
        file_path: str,
        ref: str = "main",
        project_id: str | None = None,
    ) -> str:
        """Return raw file content at *ref*."""

    # ---- Diffs -----------------------------------------------------------

    @abstractmethod
    def get_diff(
        self,
        before_sha: str,
        after_sha: str,
        project_id: str | None = None,
    ) -> list[DiffFile]:
        """Return list of files changed between two commits."""

    # ---- Permalink -------------------------------------------------------

    @abstractmethod
    def get_file_permalink(
        self,
        file_path: str,
        ref: str,
        start_line: int | None = None,
        end_line: int | None = None,
        project_id: str | None = None,
    ) -> str:
        """Build a stable permalink URL to a file (optionally with line range)."""

    # ---- Commits ---------------------------------------------------------

    @abstractmethod
    def get_commits(
        self,
        limit: int = 100,
        ref_name: str = "main",
        project_id: str | None = None,
    ) -> list[CommitInfo]:
        """Return recent commits."""

    # ---- Pull / Merge Requests -------------------------------------------

    @abstractmethod
    def get_pull_requests(
        self,
        state: str = "all",
        limit: int = 100,
        project_id: str | None = None,
    ) -> list[PRInfo]:
        """Return pull / merge requests."""

    # ---- Issues ----------------------------------------------------------

    @abstractmethod
    def get_issues(
        self,
        state: str = "all",
        limit: int = 100,
        project_id: str | None = None,
    ) -> list[IssueInfo]:
        """Return issues."""

    # ---- Comments --------------------------------------------------------

    @abstractmethod
    def get_pr_comments(
        self,
        pr_number: int,
        project_id: str | None = None,
    ) -> list[CommentInfo]:
        """Return comments / notes on a pull / merge request."""

    @abstractmethod
    def get_issue_comments(
        self,
        issue_number: int,
        project_id: str | None = None,
    ) -> list[CommentInfo]:
        """Return comments / notes on an issue."""


# ---------------------------------------------------------------------------
# Permalink Builder (platform-aware utility)
# ---------------------------------------------------------------------------


def build_permalink(
    provider: SCMProvider,
    base_url: str,
    repo_url: str,
    file_path: str,
    ref: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """
    Build an SCM permalink using the correct platform format.

    GitLab:  ``{base}/-/blob/{ref}/{path}#L{start}-{end}``
    GitHub:  ``{base}/blob/{ref}/{path}#L{start}-L{end}``

    Args:
        provider: The SCM platform.
        base_url: Base web URL of the repository.
        repo_url: Clone URL (fallback if *base_url* is empty).
        file_path: Relative path to the file in the repository.
        ref: Git ref (commit SHA preferred for stability).
        start_line: First line number (1-indexed, optional).
        end_line: Last line number (1-indexed, optional).

    Returns:
        Full permalink URL, or empty string if inputs are insufficient.
    """
    base = base_url or repo_url.rstrip("/")
    if not base:
        return ""

    if provider == SCMProvider.GITLAB:
        url = f"{base}/-/blob/{ref}/{file_path}"
        if start_line and end_line and start_line != end_line:
            url += f"#L{start_line}-{end_line}"
        elif start_line:
            url += f"#L{start_line}"
    else:
        # GitHub format
        url = f"{base}/blob/{ref}/{file_path}"
        if start_line and end_line and start_line != end_line:
            url += f"#L{start_line}-L{end_line}"
        elif start_line:
            url += f"#L{start_line}"

    return url


# ---------------------------------------------------------------------------
# Shared Git Operations (DRY helper for clone/pull)
# ---------------------------------------------------------------------------


def git_clone_or_pull(
    clone_url: str,
    local_path: str,
    branch: str,
    *,
    log_context: dict[str, Any] | None = None,
) -> str:
    """
    Clone a repository or pull the latest changes.

    Both :class:`GitLabClient` and :class:`GitHubClient` delegate their
    ``clone_or_pull`` implementations to this shared function so the
    git subprocess logic lives in exactly one place.

    Args:
        clone_url: Authenticated HTTPS clone URL.
        local_path: Absolute directory to clone into.
        branch: Branch to checkout / pull.
        log_context: Optional dict of extra fields for structured logs.

    Returns:
        *local_path* (unchanged — for convenience).

    Raises:
        subprocess.CalledProcessError: If any git command fails.
    """
    import subprocess
    from pathlib import Path

    import structlog

    _log = structlog.get_logger(__name__)
    ctx = log_context or {}

    if os.path.exists(os.path.join(local_path, ".git")):
        _log.info("pulling_repo", path=local_path, branch=branch, **ctx)
        subprocess.run(
            ["git", "-C", local_path, "fetch", "--all"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", local_path, "checkout", branch],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", local_path, "pull", "origin", branch],
            check=True, capture_output=True, text=True,
        )
    else:
        _log.info("cloning_repo", path=local_path, **ctx)
        Path(local_path).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--branch", branch, clone_url, local_path],
            check=True, capture_output=True, text=True,
        )

    _log.info("repo_ready", path=local_path, **ctx)
    return local_path


# ---------------------------------------------------------------------------
# Factory: create the right client from env / explicit provider
# ---------------------------------------------------------------------------


def get_scm_client(
    provider: SCMProvider | str | None = None,
    **kwargs: Any,
) -> SCMClient:
    """
    Factory that returns the appropriate ``SCMClient`` implementation.

    The *provider* can be:
    - ``"gitlab"`` / ``SCMProvider.GITLAB`` → ``GitLabClient``
    - ``"github"`` / ``SCMProvider.GITHUB`` → ``GitHubClient``
    - ``None`` → auto-detect from ``SCM_PROVIDER`` env var (default ``"gitlab"``)

    Any extra ``**kwargs`` are forwarded to the concrete client constructor.
    """
    import os

    if provider is None:
        provider = os.getenv("SCM_PROVIDER", "gitlab").lower()
    if isinstance(provider, str):
        provider = SCMProvider(provider)

    if provider == SCMProvider.GITLAB:
        from services.ingestion.gitlab_client import GitLabClient

        return GitLabClient(**kwargs)

    if provider == SCMProvider.GITHUB:
        from services.ingestion.github_client import GitHubClient

        return GitHubClient(**kwargs)

    raise ValueError(f"Unsupported SCM provider: {provider}")
