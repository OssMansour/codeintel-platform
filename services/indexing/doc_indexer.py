"""
CodeIntel Platform — Static Document Indexer
Indexes Application_documentation.md and incident_reports.md into Qdrant.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic_settings import BaseSettings

from services.indexing.embedder import get_embedder
from services.indexing.qdrant_store import get_store
from services.parsing.chunker import make_content_hash, make_doc_chunk_id

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class DocIndexSettings(BaseSettings):
    """Document indexer settings loaded from environment."""

    app_docs_path: str = "/data/sources/Application_documentation.md"
    incident_reports_path: str = "/data/sources/incident_reports.md"
    qdrant_collection_app_docs: str = "app_docs"
    qdrant_collection_incidents: str = "incident_reports"
    embed_batch_size: int = 32

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class DocChunk:
    """A single chunk extracted from a Markdown document."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadingSection:
    """A section of a Markdown document bounded by headings."""

    heading: str
    level: int
    section_path: str
    content: str
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Markdown Chunker
# ---------------------------------------------------------------------------


class MarkdownChunker:
    """
    Chunks Markdown documents by heading hierarchy.

    Splits at H1→H2→H3 boundaries, maintaining the heading path as metadata.
    Falls back to paragraph-based chunking if no headings are found.

    Overlap is provided by including the parent heading context in each chunk.
    """

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    MIN_CHUNK_SIZE = 50  # characters
    MAX_CHUNK_SIZE = 4000  # characters (keep well within 8192 token limit)

    def chunk_markdown(
        self,
        content: str,
        source_label: str,
        file_path: str,
        trust_level: str = "documentation",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocChunk]:
        """
        Split Markdown content into chunks.

        Chunks at heading boundaries (H1→H2→H3). Includes the heading
        hierarchy as metadata for contextual filtering.

        Args:
            content: Raw Markdown text.
            source_label: Label for the source (e.g., 'application_docs').
            file_path: Path to the source file.
            trust_level: Trust level string for Qdrant metadata.
            extra_metadata: Additional metadata fields to include.

        Returns:
            List of DocChunk objects ready for embedding and upsert.
        """
        sections = self._split_by_headings(content)

        if not sections:
            # No headings found — fallback to paragraph chunking
            log.info("no_headings_found_using_paragraph_chunking", file_path=file_path)
            sections = self._split_by_paragraphs(content)

        chunks: list[DocChunk] = []
        now = datetime.now(timezone.utc).isoformat()

        for i, section in enumerate(sections):
            chunk_content = self._build_chunk_content(section)

            if len(chunk_content.strip()) < self.MIN_CHUNK_SIZE:
                continue

            # If chunk is too large, split it further
            sub_chunks = self._maybe_split_large_chunk(
                chunk_content, section, i
            )

            for j, sub_content in enumerate(sub_chunks):
                content_hash = make_content_hash(sub_content)
                chunk_idx = i * 100 + j  # Stable index
                chunk_id = make_doc_chunk_id(
                    source_label, file_path, chunk_idx, sub_content
                )

                metadata: dict[str, Any] = {
                    "source_label": source_label,
                    "file_path": file_path,
                    "section_path": section.section_path,
                    "heading": section.heading,
                    "chunk_index": chunk_idx,
                    "content_hash": content_hash,
                    "source_collection": self._collection_for(source_label),
                    "trust_level": trust_level,
                    "last_indexed_at": now,
                }

                if extra_metadata:
                    metadata.update(extra_metadata)

                # Extract severity/date from incident headings
                if source_label == "incident_reports":
                    metadata.update(self._extract_incident_metadata(section.heading))

                chunks.append(
                    DocChunk(id=chunk_id, content=sub_content, metadata=metadata)
                )

        return chunks

    def _split_by_headings(self, content: str) -> list[HeadingSection]:
        """Split Markdown content into sections at heading boundaries."""
        lines = content.split("\n")
        sections: list[HeadingSection] = []
        heading_stack: list[tuple[int, str]] = []  # (level, heading)

        current_heading = ""
        current_level = 0
        current_content_lines: list[str] = []
        current_start = 0

        def flush_section(end_line: int) -> None:
            nonlocal current_content_lines, current_heading, current_level, current_start
            if current_content_lines or current_heading:
                section_path = " / ".join(h for _, h in heading_stack)
                sections.append(
                    HeadingSection(
                        heading=current_heading,
                        level=current_level,
                        section_path=section_path,
                        content="\n".join(current_content_lines).strip(),
                        start_line=current_start,
                        end_line=end_line,
                    )
                )
            current_content_lines = []

        for line_num, line in enumerate(lines):
            match = self.HEADING_PATTERN.match(line)
            if match:
                flush_section(line_num)
                level = len(match.group(1))
                heading_text = match.group(2).strip()

                # Maintain heading stack
                heading_stack = [(l, h) for l, h in heading_stack if l < level]
                heading_stack.append((level, heading_text))

                current_heading = heading_text
                current_level = level
                current_start = line_num + 1
            else:
                current_content_lines.append(line)

        flush_section(len(lines))

        return [s for s in sections if s.content.strip() or s.heading]

    def _split_by_paragraphs(self, content: str) -> list[HeadingSection]:
        """Fallback: split by blank lines into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", content)
        sections = []
        for i, para in enumerate(paragraphs):
            stripped = para.strip()
            if stripped:
                sections.append(
                    HeadingSection(
                        heading=f"Paragraph {i+1}",
                        level=0,
                        section_path=f"Paragraph {i+1}",
                        content=stripped,
                        start_line=i,
                        end_line=i,
                    )
                )
        return sections

    def _build_chunk_content(self, section: HeadingSection) -> str:
        """Build the full text content for a chunk including the heading."""
        parts = []
        if section.heading:
            prefix = "#" * max(section.level, 1)
            parts.append(f"{prefix} {section.heading}")
        if section.content:
            parts.append(section.content)
        return "\n\n".join(parts)

    def _maybe_split_large_chunk(
        self,
        content: str,
        section: HeadingSection,
        base_idx: int,
    ) -> list[str]:
        """
        Split an oversized chunk by paragraphs if it exceeds MAX_CHUNK_SIZE.

        Returns a list of content strings (may be just [content] if not too large).
        """
        if len(content) <= self.MAX_CHUNK_SIZE:
            return [content]

        # Split by paragraphs within the section
        paragraphs = re.split(r"\n\s*\n", section.content)
        result_chunks = []
        current = f"# {section.heading}\n\n" if section.heading else ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) > self.MAX_CHUNK_SIZE and current.strip():
                result_chunks.append(current.strip())
                current = f"# {section.heading} (continued)\n\n{para}\n\n"
            else:
                current += para + "\n\n"

        if current.strip():
            result_chunks.append(current.strip())

        return result_chunks if result_chunks else [content[: self.MAX_CHUNK_SIZE]]

    @staticmethod
    def _collection_for(source_label: str) -> str:
        """Map source label to collection name."""
        mapping = {
            "application_docs": "app_docs",
            "incident_reports": "incident_reports",
            "generated": "app_docs",
        }
        return mapping.get(source_label, "app_docs")

    @staticmethod
    def _extract_incident_metadata(heading: str) -> dict[str, Any]:
        """
        Extract severity and date from an incident heading.

        Patterns like "P1 Incident 2025-03-15: Payment Queue Timeout"
        are parsed for severity (P1-P4) and date.
        """
        metadata: dict[str, Any] = {}

        # Extract severity (P1, P2, P3, P4)
        severity_match = re.search(r"\b(P[1-4])\b", heading, re.IGNORECASE)
        if severity_match:
            metadata["severity"] = severity_match.group(1).upper()

        # Extract date (YYYY-MM-DD format)
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", heading)
        if date_match:
            metadata["incident_date"] = date_match.group(1)

        # Extract affected services (words after "service", "api", or before ":")
        service_match = re.findall(
            r"\b([A-Z][a-z]+(?:Service|Api|Worker|Queue|Handler))\b",
            heading,
        )
        if service_match:
            metadata["affected_services"] = service_match

        return metadata


# ---------------------------------------------------------------------------
# Main Doc Indexer
# ---------------------------------------------------------------------------


class DocIndexer:
    """
    Indexes static Markdown documents into Qdrant.

    Handles Application_documentation.md and incident_reports.md,
    chunking them by heading hierarchy and embedding them into
    the appropriate Qdrant collections.
    """

    def __init__(self) -> None:
        """Initialize the document indexer with embedder and store singletons."""
        self._cfg = DocIndexSettings()
        self._chunker = MarkdownChunker()
        self._embedder = get_embedder()
        self._store = get_store()
        self._log = structlog.get_logger(__name__)

    def index_markdown_file(
        self,
        file_path: str,
        collection_name: str,
        source_label: str,
        trust_level: str = "documentation",
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Index a Markdown file into a Qdrant collection.

        Chunks the file by heading hierarchy, embeds each chunk,
        and upserts into the specified collection. Safe to re-run:
        stable chunk IDs ensure idempotency.

        Args:
            file_path: Path to the Markdown file to index.
            collection_name: Target Qdrant collection.
            source_label: Label for metadata (e.g., 'application_docs').
            trust_level: Trust level for metadata.
            extra_metadata: Additional metadata fields.

        Returns:
            Number of chunks indexed.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        self._log.info(
            "indexing_markdown_file",
            file_path=file_path,
            collection=collection_name,
            source_label=source_label,
        )

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if not content.strip():
            self._log.warning("markdown_file_empty", file_path=file_path)
            return 0

        # Chunk the document
        chunks = self._chunker.chunk_markdown(
            content=content,
            source_label=source_label,
            file_path=file_path,
            trust_level=trust_level,
            extra_metadata=extra_metadata,
        )

        if not chunks:
            self._log.warning("no_chunks_extracted", file_path=file_path)
            return 0

        self._log.info(
            "doc_chunks_extracted",
            file_path=file_path,
            chunk_count=len(chunks),
        )

        # Embed all chunks (no query prefix for document indexing)
        texts = [c.content for c in chunks]
        content_hashes = [c.metadata.get("content_hash", "") for c in chunks]

        vectors = self._embedder.embed_texts(
            texts=texts,
            content_hashes=content_hashes,
            use_query_prefix=False,  # Document indexing: NO prefix
        )

        # Convert DocChunks to Chunks for upsert
        from services.parsing.chunker import Chunk

        index_chunks = [
            Chunk(id=dc.id, content=dc.content, metadata=dc.metadata)
            for dc in chunks
        ]

        self._store.upsert_chunks(
            collection=collection_name,
            chunks=index_chunks,
            vectors=vectors,
        )

        self._log.info(
            "markdown_file_indexed",
            file_path=file_path,
            collection=collection_name,
            chunks_indexed=len(chunks),
        )

        return len(chunks)

    def is_collection_populated(self, collection_name: str) -> bool:
        """
        Check if a collection has any indexed content.

        Args:
            collection_name: Collection to check.

        Returns:
            True if the collection exists and has at least 1 point.
        """
        return self._store.is_collection_populated(collection_name)

    def index_app_docs(self, force: bool = False) -> int:
        """
        Index Application_documentation.md if not already populated.

        Args:
            force: If True, re-index even if collection is already populated.

        Returns:
            Number of chunks indexed (0 if skipped).
        """
        collection = self._cfg.qdrant_collection_app_docs
        file_path = self._cfg.app_docs_path

        if not force and self.is_collection_populated(collection):
            self._log.info(
                "app_docs_already_indexed_skipping",
                collection=collection,
            )
            return 0

        if not os.path.exists(file_path):
            self._log.error("app_docs_file_not_found", path=file_path)
            raise FileNotFoundError(f"Application docs not found: {file_path}")

        return self.index_markdown_file(
            file_path=file_path,
            collection_name=collection,
            source_label="application_docs",
            trust_level="documentation",
        )

    def index_incident_reports(self, force: bool = False) -> int:
        """
        Index incident_reports.md if available and not already populated.

        This method does NOT raise an error if the file does not exist.
        Absence of the file is a valid configuration (the collection is optional).

        Args:
            force: If True, re-index even if collection is already populated.

        Returns:
            Number of chunks indexed (0 if skipped or file not found).
        """
        collection = self._cfg.qdrant_collection_incidents
        file_path = self._cfg.incident_reports_path

        if not file_path:
            self._log.info("incident_reports_path_not_configured_skipping")
            return 0

        if not os.path.exists(file_path):
            self._log.info(
                "incident_reports_file_not_found_skipping",
                path=file_path,
            )
            return 0

        if not force and self.is_collection_populated(collection):
            self._log.info(
                "incident_reports_already_indexed_skipping",
                collection=collection,
            )
            return 0

        return self.index_markdown_file(
            file_path=file_path,
            collection_name=collection,
            source_label="incident_reports",
            trust_level="incident_report",
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def index_markdown_file(
    file_path: str,
    collection_name: str,
    source_label: str,
    trust_level: str = "documentation",
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    """Module-level convenience wrapper for DocIndexer.index_markdown_file."""
    indexer = DocIndexer()
    return indexer.index_markdown_file(
        file_path=file_path,
        collection_name=collection_name,
        source_label=source_label,
        trust_level=trust_level,
        extra_metadata=extra_metadata,
    )


def is_collection_populated(collection_name: str) -> bool:
    """Module-level convenience wrapper for DocIndexer.is_collection_populated."""
    return DocIndexer().is_collection_populated(collection_name)
