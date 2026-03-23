"""
CodeIntel Platform — Cross-Encoder Reranker
Reranks retrieved chunks using ms-marco-MiniLM-L6-v2 for precise relevance scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from pydantic_settings import BaseSettings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class RerankerSettings(BaseSettings):
    """Reranker configuration loaded from environment."""

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_cache_dir: str = "/data/models/reranker"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RankedChunk:
    """A reranked chunk with its relevance score and rank."""

    chunk: dict[str, Any]
    score: float
    rank: int

    @property
    def file_path(self) -> str:
        """Convenience accessor for file path from chunk metadata."""
        return self.chunk.get("file_path", self.chunk.get("metadata", {}).get("file_path", ""))

    @property
    def symbol_name(self) -> str:
        """Convenience accessor for symbol name from chunk metadata."""
        return self.chunk.get("symbol_name", self.chunk.get("metadata", {}).get("symbol_name", ""))

    @property
    def start_line(self) -> int:
        """Convenience accessor for start line."""
        return self.chunk.get("start_line", self.chunk.get("metadata", {}).get("start_line", 0))

    @property
    def collection(self) -> str:
        """Convenience accessor for source collection."""
        return self.chunk.get("collection", self.chunk.get("metadata", {}).get("source_collection", "unknown"))

    @property
    def scm_permalink(self) -> str:
        """Convenience accessor for SCM permalink (GitLab or GitHub)."""
        meta = self.chunk.get("metadata", {})
        return (
            self.chunk.get("scm_permalink", meta.get("scm_permalink", ""))
            or self.chunk.get("gitlab_permalink", meta.get("gitlab_permalink", ""))
        )

    @property
    def gitlab_permalink(self) -> str:
        """Convenience accessor for permalink.

        .. deprecated:: 1.1.0
            Use :attr:`scm_permalink` instead.
        """
        return self.scm_permalink

    @property
    def content(self) -> str:
        """Convenience accessor for chunk content."""
        return self.chunk.get("content", "")


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Cross-encoder based reranker using ms-marco-MiniLM-L6-v2.

    Reranks (query, document) pairs with a more accurate relevance score
    than bi-encoder cosine similarity. Used as a final stage after
    retrieving candidates from Qdrant.

    Model: cross-encoder/ms-marco-MiniLM-L6-v2
    Size: ~34MB
    Inference: CPU (fast — 34ms per batch of 32 pairs)
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model ID. Defaults to RERANKER_MODEL env var.
            cache_dir: Directory to cache model weights.
        """
        cfg = RerankerSettings()
        self._model_name = model_name or cfg.reranker_model
        self._cache_dir = cache_dir or cfg.reranker_cache_dir
        self._model = None
        self._log = structlog.get_logger(__name__)

    def _load_model(self) -> None:
        """Load the cross-encoder model (lazy initialization)."""
        if self._model is not None:
            return

        self._log.info("loading_reranker_model", model=self._model_name)

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                device="cpu",
                cache_folder=self._cache_dir,
            )
            self._log.info("reranker_model_loaded", model=self._model_name)
        except Exception as exc:
            self._log.error(
                "reranker_model_load_failed",
                model=self._model_name,
                error=str(exc),
            )
            raise

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """
        Rerank a list of retrieved chunks by relevance to the query.

        Scores each (query, chunk_content) pair using the cross-encoder.
        Returns the top-k highest scoring chunks with their scores and ranks.

        Args:
            query: The user's original query string.
            chunks: List of chunk dicts from search results. Each must have
                a 'content' key (or metadata.content).
            top_k: Number of top-ranked results to return (default 5).

        Returns:
            List of RankedChunk objects sorted by score (descending),
            limited to top_k results.
        """
        if not chunks:
            return []

        self._load_model()

        # Build (query, content) pairs for batch scoring
        contents = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                content = chunk.get("metadata", {}).get("content", "")
            # Truncate long content to avoid model input limits
            if len(content) > 2048:
                content = content[:2048]
            contents.append(content)

        pairs = [(query, c) for c in contents]

        self._log.debug(
            "reranking_chunks",
            query=query[:80],
            total_chunks=len(chunks),
            top_k=top_k,
        )

        try:
            # Batch prediction for efficiency
            scores = self._model.predict(
                pairs,
                batch_size=min(32, len(pairs)),
                show_progress_bar=False,
            )
        except Exception as exc:
            self._log.error("reranker_predict_failed", error=str(exc))
            # Fall back to original order if reranker fails
            return [
                RankedChunk(chunk=c, score=c.get("score", 0.0), rank=i + 1)
                for i, c in enumerate(chunks[:top_k])
            ]

        # Pair scores with chunks and sort
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        result = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            result.append(
                RankedChunk(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                )
            )

        self._log.debug(
            "reranking_complete",
            top_score=result[0].score if result else 0.0,
            bottom_score=result[-1].score if result else 0.0,
        )

        return result

    def rerank_with_diversity(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
        max_per_file: int = 2,
    ) -> list[RankedChunk]:
        """
        Rerank with diversity constraint: limit results from the same file.

        Prevents the top-k from being dominated by many chunks from a single
        large file that happens to be highly relevant.

        Args:
            query: User query string.
            chunks: List of chunk dicts.
            top_k: Number of results to return.
            max_per_file: Maximum chunks from the same file in top-k.

        Returns:
            List of RankedChunk objects with file diversity.
        """
        all_ranked = self.rerank(query, chunks, top_k=len(chunks))

        result: list[RankedChunk] = []
        file_counts: dict[str, int] = {}

        for ranked_chunk in all_ranked:
            file_path = ranked_chunk.file_path or ranked_chunk.chunk.get("heading", "doc")
            file_counts[file_path] = file_counts.get(file_path, 0)

            if file_counts[file_path] < max_per_file:
                result.append(ranked_chunk)
                file_counts[file_path] += 1

                if len(result) >= top_k:
                    break

        # Re-assign ranks after diversity filter
        for i, rc in enumerate(result, start=1):
            rc.rank = i

        return result


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_reranker_instance: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Return the module-level reranker singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
