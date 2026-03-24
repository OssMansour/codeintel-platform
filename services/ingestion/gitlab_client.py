"""
CodeIntel Platform — GitLab Client
Wrapper around python-gitlab providing all repository interaction methods.
Implements the SCMClient interface for GitLab platform.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gitlab
import structlog
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from services.ingestion.scm_provider import (
    CommentInfo,
    CommitInfo,
    DiffFile,
    IssueInfo,
    PRInfo,
    SCMClient,
    SCMProvider,
    git_clone_or_pull,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class GitLabSettings(BaseSettings):
    """GitLab connection settings loaded from environment."""

    gitlab_url: str = "https://gitlab.com"
    gitlab_token: str = ""
    gitlab_project_id: str = ""
    repo_clone_base: str = "/data/repos"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Data Classes — re-exported from scm_provider for backwards compatibility
# ---------------------------------------------------------------------------

# Legacy aliases pointing to the unified SCM data classes.
# DiffFile, CommitInfo, IssueInfo are re-imported from scm_provider above.
# GitLab-specific wrappers kept as thin aliases.

MRInfo = PRInfo  # GitLab "Merge Request" ↔ unified "Pull Request"


@dataclass
class MRNote:
    """A note (comment) on a Merge Request.

    .. deprecated:: 1.1.0
        Use :class:`~services.ingestion.scm_provider.CommentInfo` instead.
        Scheduled for removal in v2.0.
    """

    id: int
    author: str
    body: str
    created_at: str
    updated_at: str
    system: bool


@dataclass
class IssueNote:
    """A note (comment) on an Issue.

    .. deprecated:: 1.1.0
        Use :class:`~services.ingestion.scm_provider.CommentInfo` instead.
        Scheduled for removal in v2.0.
    """

    id: int
    author: str
    body: str
    created_at: str
    updated_at: str
    system: bool


# ---------------------------------------------------------------------------
# GitLab Client
# ---------------------------------------------------------------------------


class GitLabClient(SCMClient):
    """
    High-level GitLab client for CodeIntel ingestion operations.

    Implements the unified SCMClient interface, wrapping python-gitlab
    to provide methods needed by the indexing pipeline:
    - Repository clone/pull
    - File content retrieval
    - Diff extraction between commits
    - MR and Issue metadata retrieval
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """
        Initialize the GitLab client.

        Args:
            url: GitLab instance URL. Defaults to GITLAB_URL env var.
            token: Personal access token. Defaults to GITLAB_TOKEN env var.
            project_id: Default project ID. Defaults to GITLAB_PROJECT_ID env var.
        """
        cfg = GitLabSettings()
        self._url = url or cfg.gitlab_url
        self._token = token or cfg.gitlab_token
        self._default_project_id = project_id or cfg.gitlab_project_id
        self._clone_base = cfg.repo_clone_base

        self._gl = gitlab.Gitlab(
            url=self._url,
            private_token=self._token,
            ssl_verify=True,
        )

        self._log = structlog.get_logger(__name__)
        self._log.info("gitlab_client_initialized", url=self._url)

    # ------------------------------------------------------------------
    # SCMClient interface properties
    # ------------------------------------------------------------------

    @property
    def provider(self) -> SCMProvider:
        return SCMProvider.GITLAB

    @property
    def base_web_url(self) -> str:
        """Return the web URL for the default project."""
        if self._default_project_id:
            try:
                project = self._get_project()
                return project.web_url
            except Exception as exc:
                self._log.debug(
                    "base_web_url_api_fallback",
                    error=str(exc),
                    project_id=self._default_project_id,
                )
                return f"{self._url.rstrip('/')}/{self._default_project_id}"
        return self._url.rstrip("/")

    def _get_project(self, project_id: str | None = None) -> Any:
        """
        Retrieve a python-gitlab Project object.

        Args:
            project_id: Project ID or namespace/path. Uses default if None.

        Returns:
            python-gitlab Project object.

        Raises:
            gitlab.exceptions.GitlabGetError: If the project is not found.
        """
        pid = project_id or self._default_project_id
        return self._gl.projects.get(pid)

    def get_diff(
        self,
        before_sha: str,
        after_sha: str,
        project_id: str | None = None,
    ) -> list[DiffFile]:
        """
        Get the list of files changed between two commit SHAs.

        Args:
            before_sha: The starting commit SHA.
            after_sha: The ending commit SHA.
            project_id: Project ID. Uses default if None.

        Returns:
            List of DiffFile objects representing changed files.
        """
        project = self._get_project(project_id)
        try:
            comparison = project.repository_compare(before_sha, after_sha)
            diffs = comparison.get("diffs", [])
            return [
                DiffFile(
                    old_path=d.get("old_path", ""),
                    new_path=d.get("new_path", ""),
                    a_mode=d.get("a_mode", ""),
                    b_mode=d.get("b_mode", ""),
                    new_file=d.get("new_file", False),
                    renamed_file=d.get("renamed_file", False),
                    deleted_file=d.get("deleted_file", False),
                    diff=d.get("diff", ""),
                )
                for d in diffs
            ]
        except gitlab.exceptions.GitlabGetError as exc:
            self._log.error(
                "get_diff_failed",
                before_sha=before_sha,
                after_sha=after_sha,
                error=str(exc),
            )
            raise

    def clone_or_pull(
        self,
        project_id: str | None = None,
        local_path: str | None = None,
        branch: str = "main",
    ) -> str:
        """
        Clone the repository if it doesn't exist locally, or pull latest changes.

        Args:
            project_id: Project ID or namespace/path.
            local_path: Local directory path. Auto-derived from project_id if None.
            branch: Branch to clone/pull. Defaults to 'main'.

        Returns:
            Absolute path to the local repository directory.

        Raises:
            subprocess.CalledProcessError: If git operations fail.
        """
        pid = project_id or self._default_project_id
        if local_path is None:
            safe_name = pid.replace("/", "_").replace(" ", "_")
            local_path = os.path.join(self._clone_base, safe_name)

        project = self._get_project(pid)
        clone_url = project.http_url_to_repo.replace(
            "https://", f"https://oauth2:{self._token}@"
        )

        return git_clone_or_pull(
            clone_url, local_path, branch,
            log_context={"provider": "gitlab", "project_id": pid},
        )

    def get_file_content(
        self,
        file_path: str,
        ref: str = "main",
        project_id: str | None = None,
    ) -> str:
        """
        Retrieve the raw content of a file at a specific commit/branch.

        Args:
            file_path: Path to the file relative to repository root.
            ref: Git ref (branch, tag, or commit SHA).
            project_id: Project ID. Uses default if None.

        Returns:
            File content as a UTF-8 string.

        Raises:
            gitlab.exceptions.GitlabGetError: If the file is not found.
        """
        project = self._get_project(project_id)
        try:
            f = project.files.get(file_path=file_path, ref=ref)
            content = f.decode().decode("utf-8", errors="replace")
            self._log.debug(
                "file_content_retrieved",
                file_path=file_path,
                ref=ref,
                size=len(content),
            )
            return content
        except gitlab.exceptions.GitlabGetError as exc:
            self._log.error(
                "get_file_content_failed",
                file_path=file_path,
                ref=ref,
                error=str(exc),
            )
            raise

    def get_file_permalink(
        self,
        file_path: str,
        ref: str,
        start_line: int | None = None,
        end_line: int | None = None,
        project_id: str | None = None,
    ) -> str:
        """
        Generate a GitLab permalink to a file (optionally with line range).

        Args:
            file_path: Relative path to the file.
            ref: Git ref (commit SHA preferred for stability).
            start_line: First line number (1-indexed).
            end_line: Last line number (1-indexed).
            project_id: Project ID. Uses default if None.

        Returns:
            Full GitLab web URL to the file and optional line range.
        """
        project = self._get_project(project_id)
        base_url = f"{project.web_url}/-/blob/{ref}/{file_path}"
        if start_line is not None:
            if end_line is not None and end_line != start_line:
                base_url += f"#L{start_line}-{end_line}"
            else:
                base_url += f"#L{start_line}"
        return base_url

    def get_commits(
        self,
        limit: int = 100,
        ref_name: str = "main",
        project_id: str | None = None,
    ) -> list[CommitInfo]:
        """
        Retrieve recent commits from the repository.

        Args:
            limit: Maximum number of commits to return.
            ref_name: Branch or ref to list commits from.
            project_id: Project ID. Uses default if None.

        Returns:
            List of CommitInfo objects, most recent first.
        """
        project = self._get_project(project_id)
        commits = project.commits.list(ref_name=ref_name, get_all=False, per_page=limit)
        return [
            CommitInfo(
                sha=c.id,
                message=c.message,
                author_name=c.author_name,
                author_email=c.author_email,
                authored_date=c.authored_date,
            )
            for c in commits
        ]

    def get_mrs(
        self,
        state: str = "all",
        limit: int = 100,
        project_id: str | None = None,
    ) -> list[MRInfo]:
        """
        Retrieve Merge Requests from the repository.

        Args:
            state: MR state filter ('opened', 'closed', 'merged', 'all').
            limit: Maximum number of MRs to return.
            project_id: Project ID. Uses default if None.

        Returns:
            List of MRInfo (alias for PRInfo) objects, most recent first.
        """
        project = self._get_project(project_id)
        mrs = project.mergerequests.list(state=state, get_all=False, per_page=limit)
        return [
            MRInfo(
                id=mr.id,
                number=mr.iid,
                title=mr.title,
                description=mr.description or "",
                state=mr.state,
                author=mr.author.get("username", "") if mr.author else "",
                source_branch=mr.source_branch,
                target_branch=mr.target_branch,
                created_at=mr.created_at,
                updated_at=mr.updated_at,
                web_url=mr.web_url,
            )
            for mr in mrs
        ]

    # Unified SCM interface alias
    get_pull_requests = get_mrs

    def get_issues(
        self,
        state: str = "all",
        limit: int = 100,
        project_id: str | None = None,
    ) -> list[IssueInfo]:
        """
        Retrieve Issues from the project.

        Args:
            state: Issue state filter ('opened', 'closed', 'all').
            limit: Maximum number of issues to return.
            project_id: Project ID. Uses default if None.

        Returns:
            List of IssueInfo objects, most recent first.
        """
        project = self._get_project(project_id)
        issues = project.issues.list(state=state, get_all=False, per_page=limit)
        return [
            IssueInfo(
                id=issue.id,
                number=issue.iid,
                title=issue.title,
                description=issue.description or "",
                state=issue.state,
                author=issue.author.get("username", "") if issue.author else "",
                labels=issue.labels or [],
                created_at=issue.created_at,
                updated_at=issue.updated_at,
                web_url=issue.web_url,
            )
            for issue in issues
        ]

    def get_mr_notes(
        self,
        mr_id: int,
        project_id: str | None = None,
    ) -> list[MRNote]:
        """
        Retrieve all notes (comments) on a Merge Request.

        Args:
            mr_id: The MR IID (internal ID, as shown in the GitLab UI).
            project_id: Project ID. Uses default if None.

        Returns:
            List of MRNote objects, oldest first.
        """
        project = self._get_project(project_id)
        mr = project.mergerequests.get(mr_id)
        notes = mr.notes.list(get_all=True)
        return [
            MRNote(
                id=n.id,
                author=n.author.get("username", "") if n.author else "",
                body=n.body,
                created_at=n.created_at,
                updated_at=n.updated_at,
                system=n.system,
            )
            for n in notes
        ]

    def get_issue_notes(
        self,
        issue_id: int,
        project_id: str | None = None,
    ) -> list[IssueNote]:
        """
        Retrieve all notes (comments) on an Issue.

        Args:
            issue_id: The Issue IID (internal ID, as shown in the GitLab UI).
            project_id: Project ID. Uses default if None.

        Returns:
            List of IssueNote objects, oldest first.
        """
        project = self._get_project(project_id)
        issue = project.issues.get(issue_id)
        notes = issue.notes.list(get_all=True)
        return [
            IssueNote(
                id=n.id,
                author=n.author.get("username", "") if n.author else "",
                body=n.body,
                created_at=n.created_at,
                updated_at=n.updated_at,
                system=n.system,
            )
            for n in notes
        ]

    # ------------------------------------------------------------------
    # Unified SCMClient comment interface
    # ------------------------------------------------------------------

    def get_pr_comments(
        self,
        pr_number: int,
        project_id: str | None = None,
    ) -> list[CommentInfo]:
        """Return MR notes as unified CommentInfo objects."""
        notes = self.get_mr_notes(pr_number, project_id)
        return [
            CommentInfo(
                id=n.id,
                author=n.author,
                body=n.body,
                created_at=n.created_at,
                updated_at=n.updated_at,
                system=n.system,
            )
            for n in notes
        ]

    def get_issue_comments(
        self,
        issue_number: int,
        project_id: str | None = None,
    ) -> list[CommentInfo]:
        """Return issue notes as unified CommentInfo objects."""
        notes = self.get_issue_notes(issue_number, project_id)
        return [
            CommentInfo(
                id=n.id,
                author=n.author,
                body=n.body,
                created_at=n.created_at,
                updated_at=n.updated_at,
                system=n.system,
            )
            for n in notes
        ]
