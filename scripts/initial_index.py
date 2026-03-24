#!/usr/bin/env python3
"""
CodeIntel Platform — Initial Indexing Script
One-time full repository and document indexing for platform setup.

Usage:
    python scripts/initial_index.py \\
        --project-id group/my-repo \\
        --repo-path /data/repos/my-repo \\
        --app-docs /data/sources/Application_documentation.md \\
        [--incident-reports /data/sources/incident_reports.md]

This script:
    1. Creates all 3 Qdrant collections if not exist
    2. Indexes Application_documentation.md into app_docs collection
    3. Indexes incident_reports.md into incident_reports (if provided)
    4. Walks the repository and indexes all supported code files
    5. Prints a summary report at completion
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on PYTHONPATH
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Progress tracking helper
# ---------------------------------------------------------------------------


class ProgressBar:
    """Simple terminal progress bar without external dependencies."""

    def __init__(self, total: int, description: str, width: int = 50) -> None:
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1, postfix: str = "") -> None:
        self.current = min(self.current + n, self.total)
        filled = int(self.width * self.current / max(self.total, 1))
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        sys.stdout.write(
            f"\r{self.description}: [{bar}] {self.current}/{self.total}"
            f" | {rate:.1f}/s | ETA: {int(eta)}s {postfix}"
        )
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Use tqdm if available, otherwise use our simple progress bar
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as TqdmProgress
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def make_progress(total: int, description: str) -> ProgressBar:
    """Create a progress bar (tqdm if available, else simple)."""
    if HAS_TQDM:
        return TqdmProgress(total=total, desc=description)
    return ProgressBar(total=total, description=description)


# ---------------------------------------------------------------------------
# File Walking
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    ".git", ".svn", "node_modules", "vendor", "__pycache__",
    ".venv", "venv", "env", "dist", "build", ".cache",
    ".pytest_cache", "target", ".tox",
}

LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

MAX_FILE_SIZE = int(os.getenv("PARSE_MAX_FILE_SIZE_BYTES", "1048576"))  # 1MB default


def walk_repo(repo_path: str) -> list[str]:
    """Walk a repository and return all supported source file paths (relative)."""
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in LANGUAGE_EXTENSIONS:
                continue
            full_path = os.path.join(root, fname)
            try:
                if os.path.getsize(full_path) > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            rel_path = os.path.relpath(full_path, repo_path)
            files.append(rel_path)
    return sorted(files)


# ---------------------------------------------------------------------------
# Main Indexing Logic
# ---------------------------------------------------------------------------


def setup_collections() -> None:
    """Initialize all 3 Qdrant collections."""
    from services.indexing.qdrant_store import get_store
    store = get_store()
    store.setup_collections()
    print("✓ Qdrant collections ready: code_repo, app_docs, incident_reports")


def index_app_docs(app_docs_path: str, force: bool = False) -> int:
    """Index Application_documentation.md into app_docs collection."""
    from services.indexing.doc_indexer import DocIndexer
    indexer = DocIndexer()

    if not os.path.exists(app_docs_path):
        print(f"✗ ERROR: Application docs not found: {app_docs_path}")
        sys.exit(1)

    print(f"\n[2/4] Indexing Application_documentation.md...")
    count = indexer.index_app_docs(force=force)
    if count == 0 and not force:
        print("  ↳ Already indexed (use --force to re-index)")
    else:
        print(f"  ↳ Indexed {count} chunks into app_docs collection")
    return count


def index_incidents(incident_reports_path: str | None, force: bool = False) -> int:
    """Index incident_reports.md if provided."""
    from services.indexing.doc_indexer import DocIndexer
    indexer = DocIndexer()

    if not incident_reports_path:
        print("\n[3/4] Incident reports: not provided (skipping)")
        return 0

    if not os.path.exists(incident_reports_path):
        print(f"\n[3/4] Incident reports: file not found at {incident_reports_path} (skipping)")
        return 0

    print(f"\n[3/4] Indexing incident_reports.md...")
    count = indexer.index_incident_reports(force=force)
    if count == 0 and not force:
        print("  ↳ Already indexed (use --force to re-index)")
    else:
        print(f"  ↳ Indexed {count} chunks into incident_reports collection")
    return count


def index_repository(
    project_id: str,
    repo_path: str,
    commit_sha: str,
    repo_url: str = "",
    branch: str = "main",
) -> dict:
    """Index all code files in the repository."""
    from services.indexing.embedder import get_embedder
    from services.indexing.qdrant_store import get_store
    from services.parsing.chunker import chunk_file
    from services.parsing.parser import parse_file

    embedder = get_embedder()
    store = get_store()

    if not os.path.exists(repo_path):
        print(f"✗ ERROR: Repository path not found: {repo_path}")
        sys.exit(1)

    print(f"\n[4/4] Indexing repository: {project_id}")
    print(f"       Path: {repo_path}")

    all_files = walk_repo(repo_path)
    total_files = len(all_files)

    if total_files == 0:
        print(f"  ↳ WARNING: No supported files found in {repo_path}")
        return {"files": 0, "symbols": 0, "errors": 0}

    print(f"  ↳ Found {total_files} files to index")
    print()

    stats = {
        "files_processed": 0,
        "files_failed": 0,
        "symbols_indexed": 0,
        "cache_hits": 0,
        "errors": [],
    }

    progress = make_progress(total_files, "  Indexing")

    for file_rel_path in all_files:
        abs_path = os.path.join(repo_path, file_rel_path)
        postfix = ""

        try:
            # Delete any existing stale chunks
            store.delete_by_file(
                "code_repo",
                file_rel_path,
                project_id,
            )

            # Parse the file
            symbol_table = parse_file(abs_path)
            # ⚑ Normalise: store the repo-relative path, not the OS absolute
            # path.  Without this fix, metadata["file_path"] would be e.g.
            # "C:\codeintel-data\repos\MyRepo\src\foo.py" and the SCM permalink
            # would be malformed ("file://…/repo/-/blob/main/C:\codeintel-…").
            symbol_table.file_path = file_rel_path

            # Chunk at symbol boundaries
            chunks = chunk_file(
                symbol_table=symbol_table,
                project_id=project_id,
                repo_url=repo_url or f"file://{repo_path}",
                branch=branch,
                commit_sha=commit_sha,
            )

            if not chunks:
                stats["files_processed"] += 1
                if hasattr(progress, 'set_postfix_str'):
                    progress.set_postfix_str(f"(empty: {Path(file_rel_path).name})")
                    progress.update(1)
                else:
                    progress.update(1, f"(empty: {Path(file_rel_path).name})")
                continue

            # Embed with caching
            texts = [c.content for c in chunks]
            content_hashes = [c.metadata.get("content_hash", "") for c in chunks]

            vectors = embedder.embed_texts(
                texts=texts,
                content_hashes=content_hashes,
                use_query_prefix=False,
            )

            # Upsert to Qdrant
            store.upsert_chunks(
                collection="code_repo",
                chunks=chunks,
                vectors=vectors,
            )

            stats["files_processed"] += 1
            stats["symbols_indexed"] += len(chunks)
            postfix = f"({len(chunks)} symbols)"

        except Exception as exc:
            stats["files_failed"] += 1
            stats["errors"].append(f"{file_rel_path}: {str(exc)[:100]}")
            postfix = "(FAILED)"

        if hasattr(progress, 'set_postfix_str'):
            progress.set_postfix_str(postfix)
            progress.update(1)
        else:
            progress.update(1, postfix)

    progress.close()
    return stats


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CodeIntel Platform — Initial Repository and Document Indexer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full index with all documents:
  python scripts/initial_index.py \\
      --project-id group/my-repo \\
      --repo-path /data/repos/my-repo \\
      --app-docs /data/sources/Application_documentation.md \\
      --incident-reports /data/sources/incident_reports.md

  # Force re-index (even if already indexed):
  python scripts/initial_index.py \\
      --project-id group/my-repo \\
      --repo-path /data/repos/my-repo \\
      --app-docs /data/sources/Application_documentation.md \\
      --force

  # Index only the repository (skip docs):
  python scripts/initial_index.py \\
      --project-id group/my-repo \\
      --repo-path /data/repos/my-repo \\
      --skip-docs
        """,
    )

    parser.add_argument(
        "--project-id",
        required=True,
        help="GitLab project ID (e.g., group/repo-name)",
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Absolute path to the local repository clone",
    )
    parser.add_argument(
        "--app-docs",
        default=os.getenv("APP_DOCS_PATH", "/data/sources/Application_documentation.md"),
        help="Path to Application_documentation.md",
    )
    parser.add_argument(
        "--incident-reports",
        default=os.getenv("INCIDENT_REPORTS_PATH", ""),
        help="Path to incident_reports.md (optional)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Git branch name (default: main)",
    )
    parser.add_argument(
        "--commit-sha",
        default="HEAD",
        help="Commit SHA for this indexing run (default: HEAD)",
    )
    parser.add_argument(
        "--repo-url",
        default="",
        help="Full GitLab repository URL (for permalink generation)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-index even if collections are already populated",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip document indexing, only index code",
    )
    parser.add_argument(
        "--skip-code",
        action="store_true",
        help="Skip code indexing, only index documents",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the initial indexing script."""
    args = parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     CodeIntel Platform — Initial Indexing                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Project ID:          {args.project_id}")
    print(f"  Repository Path:     {args.repo_path}")
    print(f"  App Docs:            {args.app_docs}")
    print(f"  Incident Reports:    {args.incident_reports or 'not provided'}")
    print(f"  Branch:              {args.branch}")
    print(f"  Force Re-index:      {args.force}")
    print()

    start_time = time.time()

    # Step 1: Setup Qdrant collections
    print("[1/4] Setting up Qdrant collections...")
    try:
        setup_collections()
    except Exception as exc:
        print(f"✗ ERROR: Failed to setup Qdrant collections: {exc}")
        print("  Is Qdrant running? Check: docker compose ps")
        sys.exit(1)

    # Step 2: Index Application docs
    app_docs_count = 0
    if not args.skip_docs:
        app_docs_count = index_app_docs(args.app_docs, force=args.force)

    # Step 3: Index Incident reports
    incidents_count = 0
    if not args.skip_docs:
        incidents_count = index_incidents(
            args.incident_reports if args.incident_reports else None,
            force=args.force,
        )

    # Step 4: Index repository code
    code_stats = {"files_processed": 0, "symbols_indexed": 0, "files_failed": 0, "errors": []}
    if not args.skip_code:
        code_stats = index_repository(
            project_id=args.project_id,
            repo_path=args.repo_path,
            commit_sha=args.commit_sha,
            repo_url=args.repo_url,
            branch=args.branch,
        )

    # Summary Report
    elapsed = time.time() - start_time

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     Indexing Complete — Summary Report                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Time elapsed:            {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()
    print("  Document Collections:")
    print(f"    app_docs:              {app_docs_count} chunks indexed")
    print(f"    incident_reports:      {incidents_count} chunks indexed")
    print()
    print("  Code Repository:")
    print(f"    Files processed:       {code_stats['files_processed']}")
    print(f"    Files failed:          {code_stats['files_failed']}")
    print(f"    Symbols indexed:       {code_stats['symbols_indexed']}")
    print()

    if code_stats.get("errors"):
        print("  Errors (first 10):")
        for err in code_stats["errors"][:10]:
            print(f"    - {err}")
        print()

    if code_stats["symbols_indexed"] > 0:
        print("  ✓ Platform is ready for queries!")
        print()
        print("  Test a query:")
        print('    curl -X POST http://localhost:8001/agent/query \\')
        print('      -H "Content-Type: application/json" \\')
        print(f'      -d \'{{"query": "How is this codebase structured?", "project_id": "{args.project_id}"}}\'')
    else:
        print("  ⚠ No symbols were indexed. Check:")
        print("    - Repository path contains supported files (.py, .js, .ts, .go, .java, .c, .cpp)")
        print("    - Qdrant is running and accessible")
        print("    - Embedding model is available")

    print()


if __name__ == "__main__":
    main()
