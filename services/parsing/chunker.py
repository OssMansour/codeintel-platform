"""
CodeIntel Platform — Semantic Chunker
Converts FileSymbolTable objects into indexed Chunks with stable IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from services.ingestion.scm_provider import SCMProvider, build_permalink
from services.parsing.parser import FileSymbolTable, SymbolInfo

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Chunk Data Class
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """
    A single indexed unit of code with stable ID and full metadata.

    The chunk ID is a deterministic SHA-256 hash of the content,
    enabling upsert semantics in Qdrant (unchanged content = same ID = no-op).

    Attributes:
        id: Stable SHA-256 hash string.
        content: The text content to be embedded and indexed.
        metadata: Full metadata dict conforming to the Qdrant payload schema.
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required metadata fields are present."""
        required = {
            "project_id", "file_path", "language", "symbol_name",
            "symbol_type", "start_line", "end_line", "content_hash",
            "source_collection", "trust_level", "last_indexed_at",
        }
        missing = required - set(self.metadata.keys())
        if missing:
            log.warning(
                "chunk_missing_metadata_fields",
                chunk_id=self.id,
                missing=list(missing),
            )


# ---------------------------------------------------------------------------
# Stable Chunk ID Generation
# ---------------------------------------------------------------------------


def make_chunk_id(
    project_id: str,
    file_path: str,
    symbol_name: str,
    content: str,
) -> str:
    """
    Generate a stable, deterministic SHA-256 chunk ID.

    The ID is derived from: project_id, file_path, symbol_name, and content.
    Any change to any of these components produces a different ID.

    Formula: sha256("{project_id}:{file_path}:{symbol_name}:{content}")

    Args:
        project_id: GitLab project ID or namespace/path.
        file_path: Relative path of the file from repository root.
        symbol_name: Name of the function, class, or synthetic chunk name.
        content: Full text content of the chunk.

    Returns:
        64-character hexadecimal SHA-256 digest.
    """
    key = f"{project_id}:{file_path}:{symbol_name}:{content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def make_content_hash(content: str) -> str:
    """
    Generate a SHA-256 hash of chunk content for the embedding cache key.

    Args:
        content: Text content to hash.

    Returns:
        64-character hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def chunk_file(
    symbol_table: FileSymbolTable,
    project_id: str = "",
    repo_url: str = "",
    branch: str = "main",
    commit_sha: str = "HEAD",
    scm_provider: SCMProvider | str = SCMProvider.GITLAB,
    scm_base_url: str = "",
) -> list[Chunk]:
    """
    Convert a FileSymbolTable into a list of indexed Chunks.

    Chunks are created at function/class boundaries — never at arbitrary
    token windows. Each function and class becomes exactly one chunk.

    For files with no extractable symbols (e.g., unsupported languages
    using the fallback parser), the synthetic 'chunk_N' symbols are used.

    Args:
        symbol_table: Parsed symbol table from the tree-sitter parser.
        project_id: Project identifier (used in chunk ID and metadata).
        repo_url: Full repository URL.
        branch: Git branch name.
        commit_sha: Commit SHA for this version of the file.
        scm_provider: SCM platform (``"gitlab"`` or ``"github"``).
        scm_base_url: Base web URL for building permalinks.

    Returns:
        List of Chunk objects ready for embedding and Qdrant upsert.
    """
    chunks: list[Chunk] = []
    now = datetime.now(timezone.utc).isoformat()

    all_symbols = symbol_table.all_symbols()

    if not all_symbols:
        # Edge case: file with no parseable symbols (e.g., empty file, header-only)
        log.debug(
            "no_symbols_found",
            file_path=symbol_table.file_path,
            language=symbol_table.language,
        )
        # Create a single chunk for the entire file
        content = "".join(symbol_table.source_lines)
        if content.strip():
            chunk = _make_chunk(
                project_id=project_id,
                file_path=symbol_table.file_path,
                language=symbol_table.language,
                symbol=SymbolInfo(
                    name="__module__",
                    qualified_name="__module__",
                    symbol_type="module",
                    docstring=symbol_table.module_docstring,
                    start_line=1,
                    end_line=len(symbol_table.source_lines),
                    body=content,
                ),
                repo_url=repo_url,
                branch=branch,
                commit_sha=commit_sha,
                now=now,
                scm_provider=scm_provider,
                scm_base_url=scm_base_url,
            )
            chunks.append(chunk)
        return chunks

    for symbol in all_symbols:
        # Use the symbol body if populated, otherwise extract from source lines
        content = symbol.body
        if not content.strip():
            content = "".join(
                symbol_table.source_lines[symbol.start_line - 1 : symbol.end_line]
            )

        if not content.strip():
            log.debug(
                "empty_symbol_body_skipped",
                symbol_name=symbol.name,
                file_path=symbol_table.file_path,
            )
            continue

        chunk = _make_chunk(
            project_id=project_id,
            file_path=symbol_table.file_path,
            language=symbol_table.language,
            symbol=symbol,
            repo_url=repo_url,
            branch=branch,
            commit_sha=commit_sha,
            now=now,
            scm_provider=scm_provider,
            scm_base_url=scm_base_url,
        )
        chunks.append(chunk)

    log.debug(
        "file_chunked",
        file_path=symbol_table.file_path,
        language=symbol_table.language,
        chunk_count=len(chunks),
    )
    return chunks


def _make_chunk(
    project_id: str,
    file_path: str,
    language: str,
    symbol: SymbolInfo,
    repo_url: str,
    branch: str,
    commit_sha: str,
    now: str,
    scm_provider: SCMProvider | str = SCMProvider.GITLAB,
    scm_base_url: str = "",
) -> Chunk:
    """
    Construct a single Chunk from a SymbolInfo.

    Builds the content string, generates the stable ID, and assembles
    the full metadata payload.
    """
    # Normalize to enum for type-safe comparisons downstream
    if not isinstance(scm_provider, SCMProvider):
        scm_provider = SCMProvider(scm_provider)

    # Build the content string for embedding
    # Include symbol signature context + body for richer semantics
    content = _build_content(symbol, language)
    content_hash = make_content_hash(content)
    chunk_id = make_chunk_id(project_id, file_path, symbol.qualified_name, content)

    # Build SCM permalink (GitLab or GitHub)
    permalink = _build_permalink(
        scm_provider=scm_provider,
        scm_base_url=scm_base_url,
        repo_url=repo_url,
        file_path=file_path,
        branch=branch,
        commit_sha=commit_sha,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
    )

    metadata: dict[str, Any] = {
        # Repository context
        "project_id": project_id,
        "repo_url": repo_url,
        "branch": branch,
        "commit_sha": commit_sha,
        # File context
        "file_path": file_path,
        "language": language,
        # Symbol context
        "symbol_name": symbol.name,
        "symbol_type": symbol.symbol_type,
        "qualified_name": symbol.qualified_name,
        "parent_symbol": symbol.parent or "",
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        # Content metadata
        "content_hash": content_hash,
        "source_collection": "code_repo",
        "trust_level": "code",
        "last_indexed_at": now,
        # Optional enrichment
        "docstring": symbol.docstring or "",
        "calls": symbol.calls or [],
        "params": symbol.params or [],
        # SCM permalink — canonical key for GitLab and GitHub.
        # DEPRECATED: "gitlab_permalink" kept for backwards compatibility with
        # older indexed data and consumers. Scheduled for removal in v2.0.
        "scm_permalink": permalink,
        "gitlab_permalink": permalink,
        "scm_provider": scm_provider.value,
    }

    return Chunk(id=chunk_id, content=content, metadata=metadata)


def _build_content(symbol: SymbolInfo, language: str) -> str:
    """
    Build the text content string for a symbol to be embedded.

    The content includes the docstring (if any) prepended to the body,
    giving the embedding model rich context for retrieval.

    Note: For indexing, NO query prefix is added here. The query prefix
    ("Represent this code snippet for searching...") is added ONLY at
    search time, not at index time. This is the correct CodeRankEmbed behavior.
    """
    parts: list[str] = []

    # Prepend docstring if present for richer semantics
    if symbol.docstring:
        if language == "python":
            parts.append(f'"""\n{symbol.docstring}\n"""')
        else:
            parts.append(f"/* {symbol.docstring} */")

    parts.append(symbol.body)
    return "\n".join(parts).strip()


def _build_permalink(
    scm_provider: SCMProvider | str,
    scm_base_url: str,
    repo_url: str,
    file_path: str,
    branch: str,
    commit_sha: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    Build a permalink URL for the given file and line range.

    Delegates to :func:`services.ingestion.scm_provider.build_permalink`
    to keep platform-specific URL formatting in a single place (DRY).
    """
    ref = commit_sha if commit_sha and commit_sha != "HEAD" else branch
    provider = scm_provider if isinstance(scm_provider, SCMProvider) else SCMProvider(scm_provider)
    return build_permalink(
        provider=provider,
        base_url=scm_base_url,
        repo_url=repo_url,
        file_path=file_path,
        ref=ref,
        start_line=start_line if start_line else None,
        end_line=end_line if end_line else None,
    )


# ---------------------------------------------------------------------------
# Doc Chunk ID (for Markdown documents)
# ---------------------------------------------------------------------------


def make_doc_chunk_id(
    source_label: str,
    file_path: str,
    chunk_index: int,
    content: str,
) -> str:
    """
    Generate a stable chunk ID for a Markdown document chunk.

    Args:
        source_label: Source identifier (e.g., 'application_docs', 'incident_reports').
        file_path: Path to the source Markdown file.
        chunk_index: Sequential index of this chunk within the file.
        content: Text content of the chunk.

    Returns:
        64-character hexadecimal SHA-256 digest.
    """
    key = f"{source_label}:{file_path}:{chunk_index}:{content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
