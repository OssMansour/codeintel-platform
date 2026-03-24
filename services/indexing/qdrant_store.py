"""
CodeIntel Platform — Qdrant Vector Store
Manages all 3 Qdrant collections with hybrid dense+BM25 search.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from services.parsing.chunker import Chunk

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class QdrantSettings(BaseSettings):
    """Qdrant connection settings loaded from environment."""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_code: str = "code_repo"
    qdrant_collection_app_docs: str = "app_docs"
    qdrant_collection_incidents: str = "incident_reports"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_SIZE = 768
DISTANCE = qmodels.Distance.COSINE
SPARSE_VECTOR_NAME = "bm25"
BATCH_UPSERT_SIZE = 100


# ---------------------------------------------------------------------------
# Qdrant Store
# ---------------------------------------------------------------------------


class QdrantStore:
    """
    High-level Qdrant client for CodeIntel vector storage operations.

    Manages 3 collections:
    - code_repo: Code symbols from GitLab repositories
    - app_docs: Application documentation and generated docs
    - incident_reports: Historical incident post-mortems (optional)

    Each collection uses hybrid dense+BM25 search for best retrieval quality.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the Qdrant store client.

        Args:
            url: Qdrant server URL. Defaults to QDRANT_URL env var.
            api_key: Qdrant API key. Defaults to QDRANT_API_KEY env var.
        """
        cfg = QdrantSettings()
        self._url = url or cfg.qdrant_url
        self._api_key = api_key or cfg.qdrant_api_key or None
        self._cfg = cfg

        self._client = QdrantClient(
            url=self._url,
            api_key=self._api_key,
            timeout=60,
        )

        self._collection_code = cfg.qdrant_collection_code
        self._collection_app_docs = cfg.qdrant_collection_app_docs
        self._collection_incidents = cfg.qdrant_collection_incidents

        self._log = structlog.get_logger(__name__)
        self._log.info("qdrant_store_initialized", url=self._url)

    # -----------------------------------------------------------------------
    # Collection Setup
    # -----------------------------------------------------------------------

    def setup_collections(self) -> None:
        """
        Create all 3 Qdrant collections if they do not already exist.

        Collections created:
        - code_repo: Dense 768-dim COSINE + BM25 sparse + payload indices
        - app_docs: Dense 768-dim COSINE + BM25 sparse
        - incident_reports: Dense 768-dim COSINE + BM25 sparse

        Idempotent: safe to call multiple times.
        """
        self._create_code_repo_collection()
        self._create_app_docs_collection()
        self._create_incident_reports_collection()
        self._log.info("all_collections_ready")

    def _create_code_repo_collection(self) -> None:
        """Create the code_repo collection with full payload indices."""
        name = self._collection_code
        if self._collection_exists(name):
            self._log.info("collection_already_exists", collection=name)
            return

        # Dense-only: hybrid search uses Qdrant payload TEXT index (MatchText) for
        # lexical matching — the sparse vector slot is not needed and would waste
        # memory since upsert_chunks only writes dense vectors (GAP-08).
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE,
            ),
        )

        # Create payload indices for efficient filtering
        indices = [
            ("project_id", qmodels.PayloadSchemaType.KEYWORD),
            ("branch", qmodels.PayloadSchemaType.KEYWORD),
            ("file_path", qmodels.PayloadSchemaType.KEYWORD),
            ("language", qmodels.PayloadSchemaType.KEYWORD),
            ("symbol_name", qmodels.PayloadSchemaType.KEYWORD),
            ("symbol_type", qmodels.PayloadSchemaType.KEYWORD),
            ("parent_symbol", qmodels.PayloadSchemaType.KEYWORD),
            ("source_collection", qmodels.PayloadSchemaType.KEYWORD),
            ("trust_level", qmodels.PayloadSchemaType.KEYWORD),
            ("start_line", qmodels.PayloadSchemaType.INTEGER),
            ("last_indexed_at", qmodels.PayloadSchemaType.DATETIME),
        ]

        for field_name, schema_type in indices:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema_type,
            )

        # Full-text index on content enables BM25 keyword matching in hybrid_search
        self._client.create_payload_index(
            collection_name=name,
            field_name="content",
            field_schema=qmodels.TextIndexParams(
                type=qmodels.TextIndexType.TEXT,
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=40,
                lowercase=True,
            ),
        )

        self._log.info("collection_created", collection=name)

    def _create_app_docs_collection(self) -> None:
        """Create the app_docs collection."""
        name = self._collection_app_docs
        if self._collection_exists(name):
            self._log.info("collection_already_exists", collection=name)
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE,
            ),
        )

        indices = [
            ("source_label", qmodels.PayloadSchemaType.KEYWORD),
            ("source_collection", qmodels.PayloadSchemaType.KEYWORD),
            ("trust_level", qmodels.PayloadSchemaType.KEYWORD),
            ("project_id", qmodels.PayloadSchemaType.KEYWORD),
            ("symbol_name", qmodels.PayloadSchemaType.KEYWORD),
            ("symbol_type", qmodels.PayloadSchemaType.KEYWORD),
            ("section_path", qmodels.PayloadSchemaType.KEYWORD),  # GAP-11: index for search_app_docs filter
            ("heading", qmodels.PayloadSchemaType.KEYWORD),
            ("last_indexed_at", qmodels.PayloadSchemaType.DATETIME),
        ]

        for field_name, schema_type in indices:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema_type,
            )

        self._client.create_payload_index(
            collection_name=name,
            field_name="content",
            field_schema=qmodels.TextIndexParams(
                type=qmodels.TextIndexType.TEXT,
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=40,
                lowercase=True,
            ),
        )

        self._log.info("collection_created", collection=name)

    def _create_incident_reports_collection(self) -> None:
        """Create the incident_reports collection."""
        name = self._collection_incidents
        if self._collection_exists(name):
            self._log.info("collection_already_exists", collection=name)
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE,
            ),
        )

        indices = [
            ("source_label", qmodels.PayloadSchemaType.KEYWORD),
            ("source_collection", qmodels.PayloadSchemaType.KEYWORD),
            ("severity", qmodels.PayloadSchemaType.KEYWORD),
            ("incident_date", qmodels.PayloadSchemaType.KEYWORD),
            ("heading", qmodels.PayloadSchemaType.KEYWORD),
            ("last_indexed_at", qmodels.PayloadSchemaType.DATETIME),
        ]

        for field_name, schema_type in indices:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema_type,
            )

        self._client.create_payload_index(
            collection_name=name,
            field_name="content",
            field_schema=qmodels.TextIndexParams(
                type=qmodels.TextIndexType.TEXT,
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=40,
                lowercase=True,
            ),
        )

        self._log.info("collection_created", collection=name)

    def _collection_exists(self, name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        try:
            self._client.get_collection(name)
            return True
        except (UnexpectedResponse, Exception):
            return False

    # -----------------------------------------------------------------------
    # Upsert Operations
    # -----------------------------------------------------------------------

    def upsert_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Batch upsert chunks and their vectors into a Qdrant collection.

        Uses the stable chunk ID for upsert semantics: same ID → overwrite,
        new ID → insert. Processes in batches of BATCH_UPSERT_SIZE.

        Args:
            collection: Target collection name.
            chunks: List of Chunk objects with metadata.
            vectors: Corresponding dense embedding vectors (one per chunk).

        Raises:
            ValueError: If chunks and vectors lengths don't match.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )

        if not chunks:
            return

        points: list[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            # Qdrant requires UUID or unsigned int as point ID.
            # Convert our 64-char SHA-256 hex to a valid UUID (use first 32 hex chars).
            point_id = str(uuid.UUID(chunk.id[:32]))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk.metadata,
                )
            )

        # Process in batches
        for batch_start in range(0, len(points), BATCH_UPSERT_SIZE):
            batch = points[batch_start : batch_start + BATCH_UPSERT_SIZE]
            self._client.upsert(
                collection_name=collection,
                points=batch,
                wait=True,
            )

        self._log.info(
            "chunks_upserted",
            collection=collection,
            count=len(chunks),
        )

    # -----------------------------------------------------------------------
    # Delete Operations
    # -----------------------------------------------------------------------

    def delete_by_file(
        self,
        collection: str,
        file_path: str,
        project_id: str,
    ) -> int:
        """
        Delete all chunks belonging to a specific file and project.

        Called before re-indexing a file to ensure stale chunks are removed.

        Args:
            collection: Collection to delete from.
            file_path: Relative file path to match.
            project_id: Project ID to match.

        Returns:
            Number of points deleted.
        """
        try:
            result = self._client.delete(
                collection_name=collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="file_path",
                                match=qmodels.MatchValue(value=file_path),
                            ),
                            qmodels.FieldCondition(
                                key="project_id",
                                match=qmodels.MatchValue(value=project_id),
                            ),
                        ]
                    )
                ),
                wait=True,
            )
            self._log.info(
                "file_deleted_from_collection",
                collection=collection,
                file_path=file_path,
                project_id=project_id,
                status=str(result.status),
            )
            # Qdrant UpdateResult has no deleted_count; return 1 on success, 0 otherwise.
            from qdrant_client.http.models import UpdateStatus
            return 1 if result.status == UpdateStatus.COMPLETED else 0
        except Exception as exc:
            self._log.error(
                "delete_by_file_failed",
                collection=collection,
                file_path=file_path,
                error=str(exc),
            )
            raise

    def delete_by_project(self, collection: str, project_id: str) -> None:
        """
        Delete all chunks for an entire project from a collection.

        Args:
            collection: Collection to delete from.
            project_id: Project ID to match.
        """
        self._client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="project_id",
                            match=qmodels.MatchValue(value=project_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        self._log.info(
            "project_deleted_from_collection",
            collection=collection,
            project_id=project_id,
        )

    # -----------------------------------------------------------------------
    # Search Operations
    # -----------------------------------------------------------------------

    def hybrid_search(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid dense+BM25 search on a collection.

        Combines dense cosine similarity with BM25 lexical matching via
        Qdrant's native hybrid search (Reciprocal Rank Fusion).

        Args:
            collection: Collection to search.
            query_vector: Dense embedding vector for the query.
            query_text: Raw query text for BM25 matching.
            filters: Optional payload filter dict (e.g., {'language': 'python'}).
            top_k: Number of results to return.

        Returns:
            List of result dicts with 'chunk_id', 'score', 'content', and 'metadata'.
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        try:
            # 1. Dense vector search — retrieve 2× candidates for RRF fusion
            dense_results = self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k * 2,
                with_payload=True,
                with_vectors=False,
            )

            # 2. Full-text keyword search — uses payload MatchText on content field.
            #    Requires a TEXT index on "content" (created in setup_collections).
            #    Gracefully skipped if the index does not exist yet.
            text_results = []
            if query_text.strip():
                try:
                    text_condition = qmodels.FieldCondition(
                        key="content",
                        match=qmodels.MatchText(text=query_text),
                    )
                    if qdrant_filter and qdrant_filter.must:
                        text_filter = qmodels.Filter(
                            must=list(qdrant_filter.must) + [text_condition]
                        )
                    else:
                        text_filter = qmodels.Filter(must=[text_condition])

                    text_scroll, _ = self._client.scroll(
                        collection_name=collection,
                        scroll_filter=text_filter,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=False,
                    )
                    text_results = text_scroll
                except Exception as text_exc:
                    self._log.debug(
                        "text_search_skipped",
                        collection=collection,
                        reason=str(text_exc),
                    )

            # 3. Reciprocal Rank Fusion (RRF) — k=60 is the standard constant
            rrf_k = 60
            rrf_scores: dict[str, float] = {}
            payloads: dict[str, dict] = {}

            for rank, hit in enumerate(dense_results, start=1):
                cid = str(hit.id)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                payloads[cid] = hit.payload or {}

            for rank, hit in enumerate(text_results, start=1):
                cid = str(hit.id)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                if cid not in payloads:
                    payloads[cid] = hit.payload or {}

            sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

            results = []
            for cid in sorted_ids:
                payload = payloads[cid]
                results.append({
                    "chunk_id": cid,
                    "score": rrf_scores[cid],
                    "content": payload.get("content", ""),
                    "metadata": payload,
                    "collection": collection,
                })

            return results

        except (UnexpectedResponse, ValueError) as exc:
            err = str(exc)
            if (
                "Not found" in err
                or "doesn't exist" in err
                or "not found" in err.lower()
            ):
                self._log.warning(
                    "collection_not_found_returning_empty",
                    collection=collection,
                    error=err,
                )
                return []
            self._log.error(
                "hybrid_search_failed",
                collection=collection,
                error=err,
            )
            raise
        except Exception as exc:
            self._log.error(
                "hybrid_search_failed",
                collection=collection,
                error=str(exc),
            )
            raise

    def search_all_collections(
        self,
        query_vector: list[float],
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k_per_collection: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Fan-out search across all 3 collections simultaneously.

        The incident_reports collection is searched with graceful degradation:
        if it is empty or does not exist, an empty list is returned without error.

        Args:
            query_vector: Dense embedding vector for the query.
            query_text: Raw query text for BM25.
            filters: Optional payload filters.
            top_k_per_collection: Results to return from each collection.

        Returns:
            Combined list of results from all collections, sorted by score.
        """
        all_results: list[dict[str, Any]] = []

        # Search code_repo (always required)
        try:
            code_results = self.hybrid_search(
                self._collection_code,
                query_vector,
                query_text,
                filters=filters,
                top_k=top_k_per_collection * 2,  # More from primary collection
            )
            all_results.extend(code_results)
        except Exception as exc:
            self._log.error("code_repo_search_failed", error=str(exc))
            raise

        # Search app_docs (always required)
        try:
            doc_results = self.hybrid_search(
                self._collection_app_docs,
                query_vector,
                query_text,
                top_k=top_k_per_collection,
            )
            all_results.extend(doc_results)
        except Exception as exc:
            self._log.error("app_docs_search_failed", error=str(exc))
            raise

        # Search incident_reports (optional — graceful degradation)
        try:
            if self.is_collection_populated(self._collection_incidents):
                incident_results = self.hybrid_search(
                    self._collection_incidents,
                    query_vector,
                    query_text,
                    top_k=top_k_per_collection,
                )
                all_results.extend(incident_results)
            else:
                self._log.debug(
                    "incident_reports_collection_empty_skipping",
                    collection=self._collection_incidents,
                )
        except Exception as exc:
            self._log.warning(
                "incident_reports_search_failed_graceful_skip",
                error=str(exc),
            )
            # Do not re-raise — incident_reports is optional

        # Deduplicate by chunk_id, keeping highest score
        seen: dict[str, dict[str, Any]] = {}
        for result in all_results:
            cid = result["chunk_id"]
            if cid not in seen or result["score"] > seen[cid]["score"]:
                seen[cid] = result

        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------

    def is_collection_populated(self, collection_name: str) -> bool:
        """
        Check if a collection exists and contains at least one point.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if the collection exists and has at least 1 point.
        """
        try:
            info = self._client.get_collection(collection_name)
            return info.points_count is not None and info.points_count > 0
        except Exception:
            return False

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """
        Return statistics for a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Dict with points_count, status, and vector_config info.
        """
        try:
            info = self._client.get_collection(collection_name)
            return {
                "collection": collection_name,
                "points_count": info.points_count,
                "status": str(info.status),
                "exists": True,
            }
        except Exception:
            return {
                "collection": collection_name,
                "points_count": 0,
                "status": "not_found",
                "exists": False,
            }

    def get_all_collection_stats(self) -> dict[str, dict[str, Any]]:
        """Return statistics for all 3 collections."""
        return {
            "code_repo": self.get_collection_stats(self._collection_code),
            "app_docs": self.get_collection_stats(self._collection_app_docs),
            "incident_reports": self.get_collection_stats(self._collection_incidents),
        }

    def _build_filter(self, filters: dict[str, Any]) -> qmodels.Filter:
        """
        Build a Qdrant Filter from a simple key→value dict.

        Args:
            filters: Dict where keys are field names and values are match values.

        Returns:
            Qdrant Filter object.
        """
        conditions: list[qmodels.FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value),
                    )
                )
        return qmodels.Filter(must=conditions)

    def retrieve_by_id(
        self, collection: str, chunk_id: str
    ) -> dict[str, Any] | None:
        """
        Retrieve a single point by its chunk ID.

        Args:
            collection: Collection to retrieve from.
            chunk_id: The stable chunk ID.

        Returns:
            Point payload dict or None if not found.
        """
        # upsert_chunks stores points under UUID-format IDs (first 32 hex chars of
        # the SHA-256 chunk_id).  Apply the same conversion here so lookups match.
        try:
            lookup_id = str(uuid.UUID(chunk_id[:32])) if len(chunk_id) >= 32 else chunk_id
            results = self._client.retrieve(
                collection_name=collection,
                ids=[lookup_id],
                with_payload=True,
            )
            if results:
                return results[0].payload
            return None
        except Exception as exc:
            self._log.error(
                "retrieve_by_id_failed",
                collection=collection,
                chunk_id=chunk_id,
                error=str(exc),
            )
            return None

    def scroll_by_file(
        self, collection: str, file_path: str, project_id: str
    ) -> list[dict[str, Any]]:
        """
        Retrieve all chunks for a specific file and project.

        Args:
            collection: Collection to scroll.
            file_path: File path to filter.
            project_id: Project ID to filter.

        Returns:
            List of payload dicts for all matching points.
        """
        results = []
        offset = None

        while True:
            try:
                batch, next_offset = self._client.scroll(
                    collection_name=collection,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="file_path",
                                match=qmodels.MatchValue(value=file_path),
                            ),
                            qmodels.FieldCondition(
                                key="project_id",
                                match=qmodels.MatchValue(value=project_id),
                            ),
                        ]
                    ),
                    with_payload=True,
                    limit=100,
                    offset=offset,
                )
            except Exception as exc:
                self._log.error(
                    "scroll_by_file_failed",
                    collection=collection,
                    file_path=file_path,
                    project_id=project_id,
                    error=str(exc),
                )
                raise
            results.extend([p.payload for p in batch if p.payload])
            if next_offset is None:
                break
            offset = next_offset

        return results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store_instance: QdrantStore | None = None


def get_store() -> QdrantStore:
    """Return the module-level QdrantStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = QdrantStore()
    return _store_instance
