"""
CodeIntel Platform — Celery Tasks
Async task definitions for indexing and documentation generation.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from celery import Celery
from celery.utils.log import get_task_logger
from kombu import Queue
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

log = structlog.get_logger(__name__)
task_log = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class CelerySettings(BaseSettings):
    """Celery and task configuration loaded from environment."""

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    repo_clone_base: str = "/data/repos"
    index_base: str = "/data/indexes"
    app_docs_path: str = "/data/sources/Application_documentation.md"
    incident_reports_path: str = "/data/sources/incident_reports.md"
    qdrant_collection_code: str = "code_repo"
    qdrant_collection_app_docs: str = "app_docs"
    qdrant_collection_incidents: str = "incident_reports"
    docgen_cooldown_seconds: int = 300
    parse_max_file_size_bytes: int = 1_048_576
    parse_languages: str = "python,javascript,typescript,go,java,c,cpp,c_sharp,kotlin,php"

    @property
    def supported_languages(self) -> list[str]:
        return [lang.strip() for lang in self.parse_languages.split(",")]

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Celery App
# ---------------------------------------------------------------------------

_cfg = CelerySettings()

celery_app = Celery(
    "codeintel",
    broker=_cfg.celery_broker_url,
    backend=_cfg.celery_result_backend,
)

celery_app.conf.update(
    # Task routing
    task_queues=(
        Queue("indexing"),
        Queue("docgen"),
    ),
    task_default_queue="indexing",
    task_routes={
        "services.docgen.tasks.index_repository": {"queue": "indexing"},
        "services.docgen.tasks.index_changed_files": {"queue": "indexing"},
        "services.docgen.tasks.index_static_docs": {"queue": "indexing"},
        "services.docgen.tasks.index_mr_context": {"queue": "indexing"},
        "services.docgen.tasks.regenerate_docs": {"queue": "docgen"},
        "services.docgen.tasks.generate_wiki": {"queue": "docgen"},
        "services.docgen.tasks.generate_wiki_incremental": {"queue": "docgen"},
    },
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Time limits
    task_soft_time_limit=3600,    # 1 hour soft limit
    task_time_limit=7200,          # 2 hour hard limit
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Result expiry
    result_expires=86400,          # 24 hours
    # Beat schedule for periodic tasks
    beat_schedule={
        "refresh-static-docs-daily": {
            "task": "services.docgen.tasks.index_static_docs",
            "schedule": 86400.0,  # Every 24 hours
        },
    },
)


# ---------------------------------------------------------------------------
# Helper: Supported File Filter
# ---------------------------------------------------------------------------

LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js", ".jsx"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp"],
    "c_sharp": [".cs"],
    "kotlin": [".kt", ".kts"],
    "php": [".php"],
}


def _get_supported_extensions(cfg: CelerySettings) -> set[str]:
    """Get the set of supported file extensions based on configured languages."""
    exts: set[str] = set()
    for lang in cfg.supported_languages:
        exts.update(LANGUAGE_EXTENSIONS.get(lang, []))
    return exts


def _is_supported_file(file_path: str, cfg: CelerySettings) -> bool:
    """Check if a file has a supported extension."""
    ext = Path(file_path).suffix.lower()
    return ext in _get_supported_extensions(cfg)


def _walk_repo(repo_path: str, cfg: CelerySettings) -> list[str]:
    """
    Walk a repository directory and return paths of all supported files.

    Skips hidden directories, vendor directories, and files over the max size.
    """
    SKIP_DIRS = {
        ".git", ".svn", "node_modules", "vendor", "__pycache__",
        ".venv", "venv", "env", ".env", "dist", "build",
        ".cache", ".pytest_cache", "target",
    }
    supported_files = []

    for root, dirs, files in os.walk(repo_path):
        # Prune skipped directories in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_path)

            if not _is_supported_file(rel_path, cfg):
                continue

            try:
                if os.path.getsize(full_path) > cfg.parse_max_file_size_bytes:
                    log.debug("file_too_large_skipping", file=rel_path)
                    continue
            except OSError:
                continue

            supported_files.append(rel_path)

    return supported_files


# ---------------------------------------------------------------------------
# Task: Full Repository Index
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="services.docgen.tasks.index_repository",
)
def index_repository(
    self,
    project_id: str,
    repo_path: str,
    commit_sha: str = "HEAD",
) -> dict[str, Any]:
    """
    Full repository indexing task.

    Parses every supported source file in the repository, embeds each symbol,
    and upserts into the code_repo Qdrant collection.

    Args:
        project_id: SCM project identifier (GitLab namespace/path or GitHub owner/repo).
        repo_path: Absolute path to the local repository clone.
        commit_sha: Commit SHA for this indexing run.

    Returns:
        Dict with indexing statistics.

    Raises:
        Retries up to 3 times with 60s delay on transient failures.
    """
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_store
    from services.parsing.chunker import chunk_file
    from services.parsing.parser import parse_file

    cfg = CelerySettings()
    embedder = get_embedder()
    store = get_store()

    task_log.info(f"[index_repository] Starting full index: project={project_id}, path={repo_path}")

    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")

    # Get all supported files
    all_files = _walk_repo(repo_path, cfg)
    task_log.info(f"[index_repository] Found {len(all_files)} files to index")

    stats = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        "files_processed": 0,
        "files_failed": 0,
        "symbols_indexed": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        for file_rel_path in all_files:
            abs_path = os.path.join(repo_path, file_rel_path)

            try:
                # Delete stale chunks for this file
                store.delete_by_file(cfg.qdrant_collection_code, file_rel_path, project_id)

                # Parse the file
                symbol_table = parse_file(abs_path)

                # Chunk at symbol boundaries
                chunks = chunk_file(
                    symbol_table=symbol_table,
                    project_id=project_id,
                    repo_url=f"git://{project_id}",
                    branch="main",
                    commit_sha=commit_sha,
                )

                if not chunks:
                    continue

                # Embed (with caching)
                texts = [c.content for c in chunks]
                content_hashes = [c.metadata.get("content_hash", "") for c in chunks]

                vectors = embedder.embed_texts(
                    texts=texts,
                    content_hashes=content_hashes,
                    use_query_prefix=False,
                )

                # Upsert to Qdrant
                store.upsert_chunks(
                    collection=cfg.qdrant_collection_code,
                    chunks=chunks,
                    vectors=vectors,
                )

                stats["files_processed"] += 1
                stats["symbols_indexed"] += len(chunks)

            except Exception as exc:
                task_log.error(f"[index_repository] Failed to index {file_rel_path}: {exc}")
                stats["files_failed"] += 1
                # Continue with other files — don't let one bad file kill the whole job

    except Exception as exc:
        task_log.error(f"[index_repository] Fatal error: {exc}")
        try:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            task_log.error(f"[index_repository] Max retries exceeded for project={project_id}")
            raise

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    task_log.info(
        f"[index_repository] Completed: {stats['symbols_indexed']} symbols from "
        f"{stats['files_processed']} files ({stats['files_failed']} failed)"
    )
    return stats


# ---------------------------------------------------------------------------
# Task: Incremental File Index
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="services.docgen.tasks.index_changed_files",
)
def index_changed_files(
    self,
    project_id: str,
    changed_files: list[str],
    removed_files: list[str],
    commit_sha: str,
    repo_path: str,
    branch: str = "main",
) -> dict[str, Any]:
    """
    Incremental indexing task for changed files only.

    Called after an SCM push event (GitLab or GitHub). Processes only
    the files that were modified or added. Removes chunks for deleted files.

    Args:
        project_id: SCM project identifier (GitLab namespace/path or GitHub owner/repo).
        changed_files: List of added/modified file paths (relative to repo root).
        removed_files: List of deleted file paths.
        commit_sha: New commit SHA.
        repo_path: Absolute path to the local repository clone.
        branch: Git branch name.

    Returns:
        Dict with incremental indexing statistics.
    """
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_store
    from services.parsing.chunker import chunk_file
    from services.parsing.parser import parse_file

    cfg = CelerySettings()
    embedder = get_embedder()
    store = get_store()

    stats = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        "files_changed": len(changed_files),
        "files_removed": len(removed_files),
        "symbols_updated": 0,
        "files_failed": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    task_log.info(
        f"[index_changed_files] project={project_id}, "
        f"changed={len(changed_files)}, removed={len(removed_files)}"
    )

    try:
        # Remove chunks for deleted files
        for file_path in removed_files:
            try:
                store.delete_by_file(cfg.qdrant_collection_code, file_path, project_id)
                task_log.info(f"[index_changed_files] Removed chunks for deleted file: {file_path}")
            except Exception as exc:
                task_log.error(f"[index_changed_files] Failed to delete {file_path}: {exc}")

        # Process changed/added files
        changed_symbols: list[str] = []

        for file_rel_path in changed_files:
            if not _is_supported_file(file_rel_path, cfg):
                task_log.debug(f"[index_changed_files] Skipping unsupported file: {file_rel_path}")
                continue

            abs_path = os.path.join(repo_path, file_rel_path)

            if not os.path.exists(abs_path):
                task_log.warning(f"[index_changed_files] File not found on disk: {abs_path}")
                continue

            try:
                file_size = os.path.getsize(abs_path)
                if file_size > cfg.parse_max_file_size_bytes:
                    task_log.warning(f"[index_changed_files] File too large, skipping: {file_rel_path}")
                    continue

                # Delete stale chunks
                store.delete_by_file(cfg.qdrant_collection_code, file_rel_path, project_id)

                # Parse
                symbol_table = parse_file(abs_path)

                # Chunk
                chunks = chunk_file(
                    symbol_table=symbol_table,
                    project_id=project_id,
                    repo_url=f"git://{project_id}",
                    branch=branch,
                    commit_sha=commit_sha,
                )

                if not chunks:
                    continue

                # Embed (cached)
                texts = [c.content for c in chunks]
                content_hashes = [c.metadata.get("content_hash", "") for c in chunks]

                vectors = embedder.embed_texts(
                    texts=texts,
                    content_hashes=content_hashes,
                    use_query_prefix=False,
                )

                # Upsert
                store.upsert_chunks(
                    collection=cfg.qdrant_collection_code,
                    chunks=chunks,
                    vectors=vectors,
                )

                stats["symbols_updated"] += len(chunks)
                changed_symbols.extend(
                    c.metadata.get("qualified_name", c.metadata.get("symbol_name", ""))
                    for c in chunks
                )

            except Exception as exc:
                task_log.error(f"[index_changed_files] Failed on {file_rel_path}: {exc}")
                stats["files_failed"] += 1

        # Trigger incremental wiki regen for changed files (Phase 3)
        if changed_symbols:
            try:
                generate_wiki_incremental.apply_async(
                    kwargs={
                        "project_id": project_id,
                        "repo_path": repo_path,
                        "changed_files": list(changed_files),
                        "removed_files": list(removed_files),
                        "commit_sha": commit_sha,
                    },
                    queue="docgen",
                    countdown=60,  # Wait 1 minute before regenerating
                )
            except Exception as exc:
                task_log.warning(f"[index_changed_files] Failed to dispatch incremental wiki: {exc}")

            # Also dispatch legacy symbol-level regen (best-effort)
            try:
                regenerate_docs.apply_async(
                    kwargs={
                        "project_id": project_id,
                        "changed_symbols": list(set(changed_symbols))[:50],
                    },
                    queue="docgen",
                    countdown=60,
                )
            except Exception as exc:
                task_log.warning(f"[index_changed_files] Failed to dispatch doc regen: {exc}")

    except Exception as exc:
        task_log.error(f"[index_changed_files] Fatal error: {exc}")
        try:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            task_log.error("[index_changed_files] Max retries exceeded")
            raise

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    task_log.info(
        f"[index_changed_files] Done: {stats['symbols_updated']} symbols updated"
    )
    return stats


# ---------------------------------------------------------------------------
# Task: Documentation Regeneration
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    name="services.docgen.tasks.regenerate_docs",
)
def regenerate_docs(
    self,
    project_id: str,
    changed_symbols: list[str],
) -> dict[str, Any]:
    """
    Regenerate documentation for changed code symbols.

    Triggered after incremental indexing to keep generated docs fresh.
    Only processes symbols that were actually changed — not the full repo.

    Args:
        project_id: GitLab project identifier.
        changed_symbols: List of qualified symbol names that changed.

    Returns:
        Dict with doc generation statistics.
    """
    from services.docgen.doc_generator import DocGenerator
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_store
    from services.parsing.chunker import Chunk, make_chunk_id, make_content_hash

    cfg = CelerySettings()
    store = get_store()
    embedder = get_embedder()
    doc_gen = DocGenerator()

    stats = {
        "project_id": project_id,
        "symbols_processed": 0,
        "docs_generated": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    task_log.info(
        f"[regenerate_docs] project={project_id}, symbols={len(changed_symbols)}"
    )

    try:
        # Retrieve full symbol data for each changed symbol
        symbols_data: list[dict[str, Any]] = []
        for sym_name in changed_symbols[:cfg.docgen_batch_size]:
            results = store.hybrid_search(
                collection=cfg.qdrant_collection_code,
                query_vector=embedder.embed_query(sym_name),
                query_text=sym_name,
                filters={"project_id": project_id},
                top_k=3,
            )
            if results:
                best = results[0]
                sym_data = best.get("metadata", {}).copy()
                sym_data["body"] = best.get("content", "")
                symbols_data.append(sym_data)
                stats["symbols_processed"] += 1

        if not symbols_data:
            task_log.info("[regenerate_docs] No symbol data found, skipping")
            return stats

        # Generate docs in topological order
        generated_docs = doc_gen.process_in_topological_order(symbols_data)

        # Index generated docs into app_docs collection
        for doc in generated_docs:
            if not doc.content or len(doc.content.strip()) < 10:
                continue

            content_hash = make_content_hash(doc.content)
            chunk_id = make_chunk_id(
                project_id=doc.project_id,
                file_path=doc.file_path,
                symbol_name=doc.symbol_name,
                content=doc.content,
            )

            doc_chunk = Chunk(
                id=chunk_id,
                content=doc.content,
                metadata={
                    "source_label": "generated",
                    "file_path": doc.file_path,
                    "section_path": doc.file_path,
                    "heading": doc.symbol_name,
                    "chunk_index": 0,
                    "content_hash": content_hash,
                    "source_collection": "app_docs",
                    "trust_level": "generated",
                    "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                    "project_id": doc.project_id,
                    "symbol_name": doc.symbol_name,
                    "symbol_type": doc.doc_type,
                },
            )

            vectors = embedder.embed_texts(
                texts=[doc.content],
                content_hashes=[content_hash],
                use_query_prefix=False,
            )

            store.upsert_chunks(
                collection=cfg.qdrant_collection_app_docs,
                chunks=[doc_chunk],
                vectors=vectors,
            )
            stats["docs_generated"] += 1

    except Exception as exc:
        task_log.error(f"[regenerate_docs] Error: {exc}")
        try:
            raise self.retry(exc=exc, countdown=120 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            raise

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    task_log.info(
        f"[regenerate_docs] Done: generated {stats['docs_generated']} docs"
    )
    return stats


# ---------------------------------------------------------------------------
# Task: Index MR / PR / Issue Context
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="services.docgen.tasks.index_mr_context",
)
def index_mr_context(
    self,
    project_id: str,
    context_type: str,
    context_id: int | str,
    title: str,
    body: str,
    comments: list[dict[str, str]] | None = None,
    state: str = "",
    author: str = "",
    url: str = "",
) -> dict[str, Any]:
    """
    Index a Merge Request, Pull Request, or Issue into the app_docs collection.

    Captures the description and discussion thread so the agent can reference
    MR/PR rationale and issue context when answering developer questions.

    Args:
        project_id: SCM project identifier.
        context_type: One of "merge_request", "pull_request", or "issue".
        context_id: MR/PR/Issue number (iid for GitLab, number for GitHub).
        title: Title of the MR/PR/Issue.
        body: Description / body text.
        comments: Optional list of comments, each ``{"author": ..., "body": ...}``.
        state: Current state (opened, closed, merged, etc.).
        author: Author username.
        url: Web URL for attribution.

    Returns:
        Dict with indexing statistics.
    """
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_store
    from services.parsing.chunker import Chunk, make_doc_chunk_id, make_content_hash

    cfg = CelerySettings()
    embedder = get_embedder()
    store = get_store()

    task_log.info(
        f"[index_mr_context] project={project_id}, type={context_type}, "
        f"id={context_id}, title={title[:60]}"
    )

    stats = {
        "project_id": project_id,
        "context_type": context_type,
        "context_id": context_id,
        "chunks_indexed": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        now = datetime.now(timezone.utc).isoformat()
        chunks: list[Chunk] = []

        # --- Build main content from title + body --------------------------
        label = context_type.replace("_", " ").title()
        main_content = f"# {label} #{context_id}: {title}\n\n"
        if state:
            main_content += f"**State:** {state}\n"
        if author:
            main_content += f"**Author:** {author}\n"
        main_content += "\n"
        main_content += body or "(no description)"

        content_hash = make_content_hash(main_content)
        chunk_id = make_doc_chunk_id(
            source_label=context_type,
            file_path=f"{context_type}/{context_id}",
            chunk_index=0,
            content=main_content,
        )

        chunks.append(Chunk(
            id=chunk_id,
            content=main_content,
            metadata={
                "source_label": context_type,
                "source_collection": "app_docs",
                "trust_level": "documentation",
                "heading": f"{label} #{context_id}: {title}",
                "section_path": f"{context_type}/{context_id}",
                "project_id": project_id,
                "content_hash": content_hash,
                "last_indexed_at": now,
                "context_type": context_type,
                "context_id": str(context_id),
                "state": state,
                "author": author,
                "url": url,
                # Satisfy chunk metadata validation
                "file_path": f"{context_type}/{context_id}",
                "language": "markdown",
                "symbol_name": f"{context_type}_{context_id}",
                "symbol_type": context_type,
                "start_line": 1,
                "end_line": main_content.count("\n") + 1,
            },
        ))

        # --- Build chunks for comments if provided -------------------------
        for idx, comment in enumerate(comments or []):
            comment_body = comment.get("body", "").strip()
            if not comment_body or len(comment_body) < 10:
                continue

            comment_author = comment.get("author", "unknown")
            comment_content = (
                f"## Comment by {comment_author} on {label} #{context_id}\n\n"
                f"{comment_body}"
            )

            c_hash = make_content_hash(comment_content)
            c_id = make_doc_chunk_id(
                source_label=context_type,
                file_path=f"{context_type}/{context_id}/comment",
                chunk_index=idx + 1,
                content=comment_content,
            )

            chunks.append(Chunk(
                id=c_id,
                content=comment_content,
                metadata={
                    "source_label": f"{context_type}_comment",
                    "source_collection": "app_docs",
                    "trust_level": "documentation",
                    "heading": f"Comment on {label} #{context_id}",
                    "section_path": f"{context_type}/{context_id}/comments",
                    "project_id": project_id,
                    "content_hash": c_hash,
                    "last_indexed_at": now,
                    "context_type": f"{context_type}_comment",
                    "context_id": str(context_id),
                    "author": comment_author,
                    "url": url,
                    "file_path": f"{context_type}/{context_id}/comment_{idx}",
                    "language": "markdown",
                    "symbol_name": f"{context_type}_{context_id}_comment_{idx}",
                    "symbol_type": "comment",
                    "start_line": 1,
                    "end_line": comment_content.count("\n") + 1,
                },
            ))

        if not chunks:
            task_log.info("[index_mr_context] No indexable content, skipping")
            return stats

        # --- Embed and upsert ----------------------------------------------
        texts = [c.content for c in chunks]
        hashes = [c.metadata.get("content_hash", "") for c in chunks]

        vectors = embedder.embed_texts(
            texts=texts,
            content_hashes=hashes,
            use_query_prefix=False,
        )

        store.upsert_chunks(
            collection=cfg.qdrant_collection_app_docs,
            chunks=chunks,
            vectors=vectors,
        )

        stats["chunks_indexed"] = len(chunks)

    except Exception as exc:
        task_log.error(f"[index_mr_context] Error: {exc}")
        try:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            task_log.error("[index_mr_context] Max retries exceeded")
            raise

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    task_log.info(
        f"[index_mr_context] Done: {stats['chunks_indexed']} chunks indexed "
        f"for {context_type} #{context_id}"
    )
    return stats


# ---------------------------------------------------------------------------
# Task: Static Document Indexing
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    name="services.docgen.tasks.index_static_docs",
)
def index_static_docs(self) -> dict[str, Any]:
    """
    Index or re-index static Markdown documents.

    Indexes Application_documentation.md into app_docs collection and
    incident_reports.md into incident_reports collection (if file exists).

    Safe to call multiple times: stable chunk IDs ensure idempotency.
    Re-indexes if content has changed (new chunk IDs for changed sections).

    Returns:
        Dict with indexing results for each document.
    """
    from services.indexing.doc_indexer import DocIndexer

    indexer = DocIndexer()
    results: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    task_log.info("[index_static_docs] Starting static document indexing")

    try:
        # Index Application_documentation.md (always required)
        try:
            app_docs_count = indexer.index_app_docs(force=True)
            results["app_docs_chunks"] = app_docs_count
            task_log.info(f"[index_static_docs] app_docs: {app_docs_count} chunks indexed")
        except FileNotFoundError as exc:
            task_log.error(f"[index_static_docs] app_docs file not found: {exc}")
            results["app_docs_error"] = str(exc)
        except Exception as exc:
            task_log.error(f"[index_static_docs] app_docs indexing failed: {exc}")
            results["app_docs_error"] = str(exc)

        # Index incident_reports.md (optional — graceful failure)
        try:
            incidents_count = indexer.index_incident_reports(force=True)
            results["incident_reports_chunks"] = incidents_count
            task_log.info(f"[index_static_docs] incident_reports: {incidents_count} chunks")
        except Exception as exc:
            task_log.warning(f"[index_static_docs] incident_reports failed (non-fatal): {exc}")
            results["incident_reports_note"] = "Skipped (file not found or not configured)"

    except Exception as exc:
        task_log.error(f"[index_static_docs] Fatal error: {exc}")
        try:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            raise

    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    return results


# ---------------------------------------------------------------------------
# Task: Hierarchical Wiki Generation
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=120,
    name="services.docgen.tasks.generate_wiki",
)
def generate_wiki(
    self,
    project_id: str,
    repo_path: str,
    repo_name: str | None = None,
    commit_sha: str = "HEAD",
    docs_dir: str | None = None,
) -> dict[str, Any]:
    """
    Full hierarchical wiki generation task.

    Runs the CodeWiki-inspired pipeline:
    1. Parse repo into components
    2. Build dependency graph (with Tarjan cycle resolution)
    3. Cluster components into modules via LLM
    4. Generate docs recursively (leaves first)
    5. Generate repo overview with Mermaid diagrams
    6. Index generated docs into app_docs collection

    Args:
        project_id: GitLab project identifier.
        repo_path: Absolute path to the local repository clone.
        repo_name: Human-readable name (defaults to directory name).
        commit_sha: Commit SHA for metadata.
        docs_dir: Override output directory.

    Returns:
        Dict with generation statistics.
    """
    from services.llm import get_provider
    from services.docgen.async_wiki_generator import AsyncWikiGenerator
    from services.docgen.wiki_generator import WikiGenConfig
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_store
    from services.parsing.chunker import Chunk, make_chunk_id, make_content_hash

    cfg = CelerySettings()
    llm = get_provider()
    embedder = get_embedder()
    store = get_store()

    wiki_cfg = WikiGenConfig(
        docs_dir=docs_dir or os.path.join(cfg.index_base, "wiki", project_id),
    )

    task_log.info(
        f"[generate_wiki] Starting wiki generation: project={project_id}, "
        f"provider={llm.provider_name}, model={llm.model_name}"
    )

    stats: dict[str, Any] = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        import asyncio

        generator = AsyncWikiGenerator(llm=llm, config=wiki_cfg)
        result = asyncio.run(generator.generate(
            repo_path=repo_path,
            project_id=project_id,
            repo_name=repo_name,
        ))

        stats.update(result.stats)

        # Index generated docs into app_docs collection (batched)
        all_chunks: list[Chunk] = []
        all_hashes: list[str] = []
        now_ts = datetime.now(timezone.utc).isoformat()

        for mod_doc in result.modules:
            if not mod_doc.content or len(mod_doc.content.strip()) < 20:
                continue

            content_hash = make_content_hash(mod_doc.content)
            chunk_id = make_chunk_id(
                project_id=project_id,
                file_path=f"wiki/{mod_doc.module_name}",
                symbol_name=mod_doc.module_name,
                content=mod_doc.content,
            )

            all_chunks.append(Chunk(
                id=chunk_id,
                content=mod_doc.content,
                metadata={
                    "source_label": "generated_wiki",
                    "file_path": f"wiki/{mod_doc.module_name}.md",
                    "section_path": mod_doc.module_name,
                    "heading": mod_doc.module_name,
                    "chunk_index": 0,
                    "content_hash": content_hash,
                    "source_collection": "app_docs",
                    "trust_level": "generated",
                    "last_indexed_at": now_ts,
                    "project_id": project_id,
                    "symbol_name": mod_doc.module_name,
                    "symbol_type": "wiki_module",
                    "mermaid_count": mod_doc.mermaid_count,
                },
            ))
            all_hashes.append(content_hash)

        # Include overview in the same batch
        if result.overview and len(result.overview.strip()) > 20:
            ov_hash = make_content_hash(result.overview)
            ov_id = make_chunk_id(project_id, "wiki/overview", "overview", result.overview)
            all_chunks.append(Chunk(
                id=ov_id,
                content=result.overview,
                metadata={
                    "source_label": "generated_wiki",
                    "file_path": "wiki/overview.md",
                    "section_path": "overview",
                    "heading": f"{result.repo_name} Overview",
                    "chunk_index": 0,
                    "content_hash": ov_hash,
                    "source_collection": "app_docs",
                    "trust_level": "generated",
                    "last_indexed_at": now_ts,
                    "project_id": project_id,
                    "symbol_name": "overview",
                    "symbol_type": "wiki_overview",
                },
            ))
            all_hashes.append(ov_hash)

        # Batch embed + upsert (single embedding call for all wiki docs)
        if all_chunks:
            all_texts = [c.content for c in all_chunks]
            all_vectors = embedder.embed_texts(
                texts=all_texts,
                content_hashes=all_hashes,
                use_query_prefix=False,
            )
            store.upsert_chunks(
                collection=cfg.qdrant_collection_app_docs,
                chunks=all_chunks,
                vectors=all_vectors,
            )
            task_log.info(
                f"[generate_wiki] Batch indexed {len(all_chunks)} docs "
                f"({len(all_texts)} embeddings in one call)"
            )

        stats["wiki_docs_indexed"] = len(all_chunks)

    except Exception as exc:
        task_log.error(f"[generate_wiki] Error: {exc}")
        try:
            raise self.retry(exc=exc, countdown=120 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            raise

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    task_log.info(
        f"[generate_wiki] Done: {stats.get('total_modules', 0)} modules, "
        f"{stats.get('wiki_docs_indexed', 0)} indexed to app_docs"
    )
    return stats


# ---------------------------------------------------------------------------
# Task: Incremental Wiki Regeneration (Phase 3)
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=120,
    name="services.docgen.tasks.generate_wiki_incremental",
)
def generate_wiki_incremental(
    self,
    project_id: str,
    repo_path: str,
    changed_files: list[str],
    removed_files: list[str] | None = None,
    commit_sha: str = "HEAD",
    repo_name: str | None = None,
    docs_dir: str | None = None,
) -> dict[str, Any]:
    """
    Incremental wiki regeneration: only re-generates docs for modules
    that contain components from changed files.

    Falls back to a full generation if no previous module_tree.json exists.

    Args:
        project_id: GitLab project identifier.
        repo_path: Absolute path to the local repository clone.
        changed_files: Relative paths of added/modified files.
        removed_files: Relative paths of deleted files.
        commit_sha: Commit SHA for metadata.
        repo_name: Human-readable name (defaults to directory name).
        docs_dir: Override output directory.

    Returns:
        Dict with generation statistics including list of regenerated modules.
    """
    from services.llm import get_provider
    from services.docgen.incremental_generator import IncrementalWikiGenerator
    from services.docgen.wiki_generator import WikiGenConfig
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_store
    from services.parsing.chunker import Chunk, make_chunk_id, make_content_hash

    cfg = CelerySettings()
    llm = get_provider()
    embedder = get_embedder()
    store = get_store()

    wiki_cfg = WikiGenConfig(
        docs_dir=docs_dir or os.path.join(cfg.index_base, "wiki", project_id),
    )

    task_log.info(
        f"[generate_wiki_incremental] project={project_id}, "
        f"changed={len(changed_files)}, removed={len(removed_files or [])}"
    )

    stats: dict[str, Any] = {
        "project_id": project_id,
        "commit_sha": commit_sha,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        import asyncio

        generator = IncrementalWikiGenerator(llm=llm, config=wiki_cfg)
        result = asyncio.run(generator.regenerate_affected(
            repo_path=repo_path,
            project_id=project_id,
            changed_files=changed_files,
            removed_files=removed_files or [],
            repo_name=repo_name,
        ))

        stats.update(result.stats)

        # Re-index only the regenerated module docs
        regenerated = set(result.stats.get("regenerated", []))
        if not regenerated:
            task_log.info("[generate_wiki_incremental] No modules regenerated, skipping indexing")
            stats["wiki_docs_indexed"] = 0
        else:
            all_chunks: list[Chunk] = []
            all_hashes: list[str] = []
            now_ts = datetime.now(timezone.utc).isoformat()

            for mod_doc in result.modules:
                # Only index regenerated modules
                if mod_doc.module_name not in regenerated:
                    continue
                if not mod_doc.content or len(mod_doc.content.strip()) < 20:
                    continue

                content_hash = make_content_hash(mod_doc.content)
                chunk_id = make_chunk_id(
                    project_id=project_id,
                    file_path=f"wiki/{mod_doc.module_name}",
                    symbol_name=mod_doc.module_name,
                    content=mod_doc.content,
                )

                all_chunks.append(Chunk(
                    id=chunk_id,
                    content=mod_doc.content,
                    metadata={
                        "source_label": "generated_wiki",
                        "file_path": f"wiki/{mod_doc.module_name}.md",
                        "section_path": mod_doc.module_name,
                        "heading": mod_doc.module_name,
                        "chunk_index": 0,
                        "content_hash": content_hash,
                        "source_collection": "app_docs",
                        "trust_level": "generated",
                        "last_indexed_at": now_ts,
                        "project_id": project_id,
                        "symbol_name": mod_doc.module_name,
                        "symbol_type": "wiki_module",
                        "mermaid_count": mod_doc.mermaid_count,
                        "incremental": True,
                    },
                ))
                all_hashes.append(content_hash)

            # Re-index overview if it was regenerated
            top_level_touched = regenerated & set(result.module_tree.keys())
            if top_level_touched and result.overview and len(result.overview.strip()) > 20:
                ov_hash = make_content_hash(result.overview)
                ov_id = make_chunk_id(project_id, "wiki/overview", "overview", result.overview)
                all_chunks.append(Chunk(
                    id=ov_id,
                    content=result.overview,
                    metadata={
                        "source_label": "generated_wiki",
                        "file_path": "wiki/overview.md",
                        "section_path": "overview",
                        "heading": f"{result.repo_name} Overview",
                        "chunk_index": 0,
                        "content_hash": ov_hash,
                        "source_collection": "app_docs",
                        "trust_level": "generated",
                        "last_indexed_at": now_ts,
                        "project_id": project_id,
                        "symbol_name": "overview",
                        "symbol_type": "wiki_overview",
                        "incremental": True,
                    },
                ))
                all_hashes.append(ov_hash)

            if all_chunks:
                all_texts = [c.content for c in all_chunks]
                all_vectors = embedder.embed_texts(
                    texts=all_texts,
                    content_hashes=all_hashes,
                    use_query_prefix=False,
                )
                store.upsert_chunks(
                    collection=cfg.qdrant_collection_app_docs,
                    chunks=all_chunks,
                    vectors=all_vectors,
                )
                task_log.info(
                    f"[generate_wiki_incremental] Indexed {len(all_chunks)} "
                    f"regenerated docs"
                )

            stats["wiki_docs_indexed"] = len(all_chunks)

    except Exception as exc:
        task_log.error(f"[generate_wiki_incremental] Error: {exc}")
        try:
            raise self.retry(exc=exc, countdown=120 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            raise

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    task_log.info(
        f"[generate_wiki_incremental] Done: "
        f"{stats.get('affected_modules', 0)} modules regenerated, "
        f"{stats.get('wiki_docs_indexed', 0)} indexed"
    )
    return stats
