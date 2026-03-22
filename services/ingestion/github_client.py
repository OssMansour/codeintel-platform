"""
CodeIntel Platform — GitHub Client
Wrapper around PyGithub providing all repository interaction methods.

Implements the same ``SCMClient`` interface as ``GitLabClient`` so the
ingestion pipeline, chunker, and webhooks work with GitHub repos.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import structlog
from github import Auth, Github
from github.GithubException import GithubException, UnknownObjectException
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


class GitHubSettings(BaseSettings):
    """GitHub connection settings loaded from environment."""

    github_url: str = "https://github.com"
    github_api_url: str = "https://api.github.com"
    github_token: str = ""
    github_repo: str = ""  # owner/repo format
    repo_clone_base: str = "/data/repos"

    class Config:
        env_file = ".env"
        case_sensitive = False


# ---------------------------------------------------------------------------
# GitHub Client
# ---------------------------------------------------------------------------


class GitHubClient(SCMClient):
    """
    High-level GitHub client for CodeIntel ingestion operations.

    Wraps PyGithub to provide methods needed by the indexing pipeline:
    - Repository clone/pull
    - File content retrieval
    - Diff extraction between commits
    - PR and Issue metadata retrieval
    """

    def __init__(
        self,
        url: str | None = None,
        api_url: str | None = None,
        token: str | None = None,
        repo: str | None = None,
    ) -> None:
        """
        Initialize the GitHub client.

        Args:
            url: GitHub web URL. Defaults to GITHUB_URL env var.
            api_url: GitHub API URL. Defaults to GITHUB_API_URL env var.
            token: Personal access token. Defaults to GITHUB_TOKEN env var.
            repo: Default repository (owner/repo). Defaults to GITHUB_REPO env var.
        """
        cfg = GitHubSettings()
        self._url = url or cfg.github_url
        self._api_url = api_url or cfg.github_api_url
        self._token = token or cfg.github_token
        self._default_repo = repo or cfg.github_repo
        self._clone_base = cfg.repo_clone_base

        auth = Auth.Token(self._token) if self._token else None
        self._gh = Github(
            base_url=self._api_url,
            auth=auth,
        )

        self._log = structlog.get_logger(__name__)
        self._log.info("github_client_initialized", url=self._url)

    # ------------------------------------------------------------------
    # SCMClient interface properties
    # ------------------------------------------------------------------

    @property
    def provider(self) -> SCMProvider:
        return SCMProvider.GITHUB

    @property
    def base_web_url(self) -> str:
        """Return the web URL for the default repository."""
        if self._default_repo:
            return f"{self._url.rstrip('/')}/{self._default_repo}"
        return self._url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_repo(self, repo: str | None = None) -> Any:
        """
        Retrieve a PyGithub Repository object.

        Args:
            repo: Repository in owner/repo format. Uses default if None.

        Returns:
            PyGithub Repository object.

        Raises:
            UnknownObjectException: If the repository is not found.
        """
        name = repo or self._default_repo
        return self._gh.get_repo(name)

    # ------------------------------------------------------------------
    # Repository operations
    # ------------------------------------------------------------------

    def clone_or_pull(
        self,
        project_id: str | None = None,
        local_path: str | None = None,
        branch: str = "main",
    ) -> str:
        """
        Clone the repository if it doesn't exist locally, or pull latest changes.

        Args:
            project_id: Repository in owner/repo format (or uses default).
            local_path: Local directory path. Auto-derived if None.
            branch: Branch to clone/pull. Defaults to 'main'.

        Returns:
            Absolute path to the local repository directory.

        Raises:
            subprocess.CalledProcessError: If git operations fail.
        """
        repo_name = project_id or self._default_repo
        if local_path is None:
            safe_name = repo_name.replace("/", "_").replace(" ", "_")
            local_path = os.path.join(self._clone_base, safe_name)

        clone_url = f"https://{self._token}@github.com/{repo_name}.git"

        return git_clone_or_pull(
            clone_url, local_path, branch,
            log_context={"provider": "github", "repo": repo_name},
        )

    # ------------------------------------------------------------------
    # Diffs
    # ------------------------------------------------------------------

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
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            List of DiffFile objects representing changed files.
        """
        repo = self._get_repo(project_id)
        try:
            comparison = repo.compare(before_sha, after_sha)
            return [
                DiffFile(
                    old_path=f.previous_filename or f.filename,
                    new_path=f.filename,
                    new_file=f.status == "added",
                    renamed_file=f.status == "renamed",
                    deleted_file=f.status == "removed",
                    diff=f.patch or "",
                )
                for f in comparison.files
            ]
        except GithubException as exc:
            self._log.error(
                "get_diff_failed",
                before_sha=before_sha,
                after_sha=after_sha,
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # File content
    # ------------------------------------------------------------------

    def get_file_content(
        self,
        file_path: str,
        ref: str = "main",
        project_id: str | None = None,
    ) -> str:
        """
        Retrieve the raw content of a file at a specific commit/branch.

        Args:
            file_path: Path relative to repository root.
            ref: Git ref (branch, tag, or commit SHA).
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            File content as a UTF-8 string.
        """
        repo = self._get_repo(project_id)
        try:
            content_file = repo.get_contents(file_path, ref=ref)
            # get_contents may return a list for directories
            if isinstance(content_file, list):
                raise ValueError(f"{file_path} is a directory, not a file")
            decoded = content_file.decoded_content.decode("utf-8", errors="replace")
            self._log.debug(
                "file_content_retrieved",
                file_path=file_path,
                ref=ref,
                size=len(decoded),
            )
            return decoded
        except (GithubException, UnknownObjectException) as exc:
            self._log.error(
                "get_file_content_failed",
                file_path=file_path,
                ref=ref,
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Permalink
    # ------------------------------------------------------------------

    def get_file_permalink(
        self,
        file_path: str,
        ref: str,
        start_line: int | None = None,
        end_line: int | None = None,
        project_id: str | None = None,
    ) -> str:
        """
        Generate a GitHub permalink to a file (optionally with line range).

        GitHub format: ``{base_url}/blob/{ref}/{path}#L{start}-L{end}``

        Args:
            file_path: Relative path to the file.
            ref: Git ref (commit SHA preferred for stability).
            start_line: First line number (1-indexed).
            end_line: Last line number (1-indexed).
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            Full GitHub web URL to the file and optional line range.
        """
        repo_name = project_id or self._default_repo
        base_url = f"{self._url.rstrip('/')}/{repo_name}/blob/{ref}/{file_path}"
        if start_line is not None:
            if end_line is not None and end_line != start_line:
                base_url += f"#L{start_line}-L{end_line}"
            else:
                base_url += f"#L{start_line}"
        return base_url

    # ------------------------------------------------------------------
    # Commits
    # ------------------------------------------------------------------

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
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            List of CommitInfo objects, most recent first.
        """
        repo = self._get_repo(project_id)
        commits = repo.get_commits(sha=ref_name)
        result: list[CommitInfo] = []
        for c in commits[:limit]:
            git_commit = c.commit
            result.append(
                CommitInfo(
                    sha=c.sha,
                    message=git_commit.message,
                    author_name=git_commit.author.name if git_commit.author else "",
                    author_email=git_commit.author.email if git_commit.author else "",
                    authored_date=(
                        git_commit.author.date.isoformat()
                        if git_commit.author and git_commit.author.date
                        else ""
                    ),
                    added=[f.filename for f in c.files if f.status == "added"] if c.files else [],
                    modified=[f.filename for f in c.files if f.status == "modified"] if c.files else [],
                    removed=[f.filename for f in c.files if f.status == "removed"] if c.files else [],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Pull Requests
    # ------------------------------------------------------------------

    def get_pull_requests(
        self,
        state: str = "all",
        limit: int = 100,
        project_id: str | None = None,
    ) -> list[PRInfo]:
        """
        Retrieve Pull Requests from the repository.

        Args:
            state: PR state filter ('open', 'closed', 'all').
            limit: Maximum number of PRs to return.
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            List of PRInfo objects, most recent first.
        """
        repo = self._get_repo(project_id)
        prs = repo.get_pulls(state=state, sort="created", direction="desc")
        result: list[PRInfo] = []
        for pr in prs[:limit]:
            result.append(
                PRInfo(
                    id=pr.id,
                    number=pr.number,
                    title=pr.title,
                    description=pr.body or "",
                    state=pr.state,
                    author=pr.user.login if pr.user else "",
                    source_branch=pr.head.ref if pr.head else "",
                    target_branch=pr.base.ref if pr.base else "",
                    created_at=pr.created_at.isoformat() if pr.created_at else "",
                    updated_at=pr.updated_at.isoformat() if pr.updated_at else "",
                    web_url=pr.html_url or "",
                )
            )
        return result

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def get_issues(
        self,
        state: str = "all",
        limit: int = 100,
        project_id: str | None = None,
    ) -> list[IssueInfo]:
        """
        Retrieve Issues from the repository.

        Args:
            state: Issue state filter ('open', 'closed', 'all').
            limit: Maximum number of issues to return.
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            List of IssueInfo objects, most recent first.
        """
        repo = self._get_repo(project_id)
        issues = repo.get_issues(state=state, sort="created", direction="desc")
        result: list[IssueInfo] = []
        for issue in issues[:limit]:
            # GitHub API returns PRs in the issues endpoint — skip them
            if issue.pull_request is not None:
                continue
            result.append(
                IssueInfo(
                    id=issue.id,
                    number=issue.number,
                    title=issue.title,
                    description=issue.body or "",
                    state=issue.state,
                    author=issue.user.login if issue.user else "",
                    labels=[l.name for l in issue.labels],
                    created_at=issue.created_at.isoformat() if issue.created_at else "",
                    updated_at=issue.updated_at.isoformat() if issue.updated_at else "",
                    web_url=issue.html_url or "",
                )
            )
            if len(result) >= limit:
                break
        return result

    # ------------------------------------------------------------------
    # PR Comments
    # ------------------------------------------------------------------

    def get_pr_comments(
        self,
        pr_number: int,
        project_id: str | None = None,
    ) -> list[CommentInfo]:
        """
        Retrieve all comments on a Pull Request.

        Includes both issue comments and review comments.

        Args:
            pr_number: The PR number.
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            List of CommentInfo objects, oldest first.
        """
        repo = self._get_repo(project_id)
        pr = repo.get_pull(pr_number)
        comments: list[CommentInfo] = []

        # Issue-level comments
        for c in pr.get_issue_comments():
            comments.append(
                CommentInfo(
                    id=c.id,
                    author=c.user.login if c.user else "",
                    body=c.body or "",
                    created_at=c.created_at.isoformat() if c.created_at else "",
                    updated_at=c.updated_at.isoformat() if c.updated_at else "",
                    system=False,
                )
            )

        # Review comments (inline)
        for rc in pr.get_review_comments():
            comments.append(
                CommentInfo(
                    id=rc.id,
                    author=rc.user.login if rc.user else "",
                    body=rc.body or "",
                    created_at=rc.created_at.isoformat() if rc.created_at else "",
                    updated_at=rc.updated_at.isoformat() if rc.updated_at else "",
                    system=False,
                )
            )

        # Sort by creation date
        comments.sort(key=lambda c: c.created_at)
        return comments

    # ------------------------------------------------------------------
    # Issue Comments
    # ------------------------------------------------------------------

    def get_issue_comments(
        self,
        issue_number: int,
        project_id: str | None = None,
    ) -> list[CommentInfo]:
        """
        Retrieve all comments on an Issue.

        Args:
            issue_number: The Issue number.
            project_id: Repository in owner/repo format. Uses default if None.

        Returns:
            List of CommentInfo objects, oldest first.
        """
        repo = self._get_repo(project_id)
        issue = repo.get_issue(issue_number)
        return [
            CommentInfo(
                id=c.id,
                author=c.user.login if c.user else "",
                body=c.body or "",
                created_at=c.created_at.isoformat() if c.created_at else "",
                updated_at=c.updated_at.isoformat() if c.updated_at else "",
                system=False,
            )
            for c in issue.get_comments()
        ]
