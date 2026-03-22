"""
CodeIntel Platform — CodeRankEmbed Embedding Service
Wraps nomic-ai/CodeRankEmbed with batching, caching, and correct query prefixing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from pydantic_settings import BaseSettings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class EmbedSettings(BaseSettings):
    """Embedding service configuration loaded from environment."""

    embed_model: str = "nomic-ai/CodeRankEmbed"
    embed_device: str = "cpu"
    embed_batch_size: int = 32
    embed_cache_dir: str = "/data/models/embed"
    embed_vector_cache_dir: str = "/data/indexes/embed_cache"

    class Config:
        env_file = ".env"
        case_sensitive = False


# ---------------------------------------------------------------------------
# Query Prefix (CodeRankEmbed specific)
# ---------------------------------------------------------------------------

# CodeRankEmbed requires this prefix for SEARCH QUERIES only.
# Do NOT add this prefix when indexing documents.
CODERERANK_QUERY_PREFIX = "Represent this code snippet for searching relevant passages: "


# ---------------------------------------------------------------------------
# Embedding Cache (file-based, survives restarts)
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """
    File-based embedding cache keyed by content hash.

    Stores embeddings as numpy .npy files so that unchanged chunks
    are never re-embedded even after service restarts.
    """

    def __init__(self, cache_dir: str) -> None:
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory where cached embeddings are stored.
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._log = structlog.get_logger(__name__)

    def _cache_path(self, content_hash: str) -> Path:
        """Return the file path for a cached embedding."""
        return self._cache_dir / f"{content_hash}.npy"

    def get(self, content_hash: str) -> list[float] | None:
        """
        Retrieve a cached embedding by content hash.

        Args:
            content_hash: SHA-256 hash of the content to look up.

        Returns:
            Embedding as a list of floats, or None if not cached.
        """
        path = self._cache_path(content_hash)
        if path.exists():
            try:
                arr = np.load(str(path))
                return arr.tolist()
            except Exception as exc:
                self._log.warning(
                    "embed_cache_read_error",
                    content_hash=content_hash,
                    error=str(exc),
                )
                return None
        return None

    def set(self, content_hash: str, embedding: list[float]) -> None:
        """
        Save an embedding to the cache.

        Args:
            content_hash: SHA-256 hash of the content.
            embedding: Embedding vector as a list of floats.
        """
        path = self._cache_path(content_hash)
        try:
            np.save(str(path), np.array(embedding, dtype=np.float32))
        except Exception as exc:
            self._log.warning(
                "embed_cache_write_error",
                content_hash=content_hash,
                error=str(exc),
            )

    def has(self, content_hash: str) -> bool:
        """Check if an embedding exists in the cache."""
        return self._cache_path(content_hash).exists()

    def invalidate(self, content_hash: str) -> None:
        """Remove a cached embedding (e.g., after content change)."""
        path = self._cache_path(content_hash)
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# CodeRankEmbed Embedder
# ---------------------------------------------------------------------------


class CodeRankEmbedder:
    """
    Embedding service using nomic-ai/CodeRankEmbed.

    Provides batched embedding of code text with:
    - Correct query/document prefix handling (query prefix for search ONLY)
    - File-based caching by content hash
    - CPU-only inference
    - Optional ONNX INT8 quantization for faster CPU inference

    Dimension: 768
    Context window: 8192 tokens
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        cache_dir: str | None = None,
        vector_cache_dir: str | None = None,
    ) -> None:
        """
        Initialize the CodeRankEmbedder.

        Args:
            model_name: HuggingFace model ID. Defaults to EMBED_MODEL env var.
            device: Device string ('cpu'). Defaults to EMBED_DEVICE env var.
            batch_size: Texts per embedding batch. Defaults to EMBED_BATCH_SIZE env var.
            cache_dir: Directory to store model weights.
            vector_cache_dir: Directory to store cached embedding vectors.
        """
        cfg = EmbedSettings()
        self._model_name = model_name or cfg.embed_model
        self._device = device or cfg.embed_device
        self._batch_size = batch_size or cfg.embed_batch_size
        self._model_dir = cache_dir or cfg.embed_cache_dir
        self._log = structlog.get_logger(__name__)
        self._model: Any | None = None
        self._cache = EmbeddingCache(vector_cache_dir or cfg.embed_vector_cache_dir)

    def _load_model(self) -> None:
        """Load the sentence-transformers model (lazy initialization)."""
        if self._model is not None:
            return

        self._log.info(
            "loading_embed_model",
            model=self._model_name,
            device=self._device,
        )

        # Try ONNX quantized model first for faster CPU inference
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._model_dir,
                trust_remote_code=True,
            )
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                self._model_name,
                export=True,
                cache_dir=self._model_dir,
            )
            self._model = _OnnxEmbedWrapper(tokenizer, ort_model)
            self._log.info("embed_model_loaded_onnx", model=self._model_name)
            return
        except (ImportError, Exception) as exc:
            self._log.info(
                "onnx_unavailable_using_sentence_transformers",
                reason=str(exc),
            )

        # Fallback: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=self._model_dir,
                trust_remote_code=True,
            )
            self._log.info(
                "embed_model_loaded_sentence_transformers",
                model=self._model_name,
                device=self._device,
            )
        except Exception as exc:
            self._log.error("embed_model_load_failed", model=self._model_name, error=str(exc))
            raise

    def embed_texts(
        self,
        texts: list[str],
        content_hashes: list[str] | None = None,
        use_query_prefix: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of texts, with optional caching and query prefix.

        For document indexing: call with use_query_prefix=False (default).
        For search queries: call with use_query_prefix=True.

        Args:
            texts: List of text strings to embed.
            content_hashes: Optional list of SHA-256 hashes (one per text).
                If provided, cached embeddings are used for cache hits.
            use_query_prefix: If True, prepend the CodeRankEmbed query prefix.
                Set True ONLY for search queries, NOT for document indexing.

        Returns:
            List of embedding vectors (one 768-dim list per input text).
        """
        if not texts:
            return []

        self._load_model()

        # Validate content_hashes length matches texts
        if content_hashes and len(content_hashes) != len(texts):
            self._log.warning(
                "embed_hash_mismatch",
                texts_count=len(texts),
                hashes_count=len(content_hashes),
            )
            content_hashes = None

        # Build result array, filling from cache where possible
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if content_hashes:
            for i, (text, h) in enumerate(zip(texts, content_hashes)):
                cached = self._cache.get(h)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = list(texts)

        self._log.debug(
            "embed_batch_info",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            to_embed=len(uncached_texts),
            use_query_prefix=use_query_prefix,
        )

        if uncached_texts:
            # Apply query prefix if needed
            if use_query_prefix:
                prefixed = [CODERERANK_QUERY_PREFIX + t for t in uncached_texts]
            else:
                prefixed = uncached_texts

            # Process in batches
            new_embeddings: list[list[float]] = []
            for batch_start in range(0, len(prefixed), self._batch_size):
                batch = prefixed[batch_start : batch_start + self._batch_size]
                batch_vecs = self._embed_batch(batch)
                new_embeddings.extend(batch_vecs)

            # Store results and update cache
            for local_idx, (global_idx, text) in enumerate(
                zip(uncached_indices, uncached_texts)
            ):
                vec = new_embeddings[local_idx]
                results[global_idx] = vec
                # Save to cache if we have a hash
                if content_hashes:
                    h = content_hashes[global_idx]
                    self._cache.set(h, vec)

        # All results must be filled
        assert all(r is not None for r in results), "Some embeddings were not computed"
        return results  # type: ignore[return-value]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query with the required CodeRankEmbed prefix.

        This is the correct method for embedding user queries at search time.
        The prefix "Represent this code snippet for searching relevant passages: "
        is automatically applied.

        Args:
            query: The search query string.

        Returns:
            768-dimensional embedding vector.
        """
        self._load_model()
        prefixed = CODERERANK_QUERY_PREFIX + query
        vecs = self._embed_batch([prefixed])
        return vecs[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Run inference on a single batch of texts.

        Args:
            texts: List of (optionally prefixed) text strings.

        Returns:
            List of 768-dimensional embedding vectors.
        """
        try:
            if hasattr(self._model, "encode"):
                # sentence-transformers SentenceTransformer
                vecs = self._model.encode(
                    texts,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return vecs.tolist()
            elif hasattr(self._model, "embed"):
                # ONNX wrapper
                return self._model.embed(texts)
            else:
                raise RuntimeError(f"Unknown model type: {type(self._model)}")
        except Exception as exc:
            self._log.error(
                "embed_batch_failed",
                batch_size=len(texts),
                error=str(exc),
            )
            raise

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (768 for CodeRankEmbed)."""
        return 768


# ---------------------------------------------------------------------------
# ONNX Wrapper
# ---------------------------------------------------------------------------


class _OnnxEmbedWrapper:
    """
    Thin wrapper around an ONNX-exported embedding model.

    Implements the same interface as sentence-transformers for compatibility.
    """

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self._tokenizer = tokenizer
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the ONNX model."""
        import torch

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy().tolist()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_embedder_instance: CodeRankEmbedder | None = None


def get_embedder() -> CodeRankEmbedder:
    """Return the module-level embedder singleton."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = CodeRankEmbedder()
    return _embedder_instance
