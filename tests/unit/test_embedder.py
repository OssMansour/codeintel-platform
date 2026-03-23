"""
Unit tests for services/indexing/embedder.py

EmbeddingCache is tested with real file I/O (using pytest tmp_path).
CodeRankEmbedder is tested with a mock model injected directly so no
sentence-transformers model is downloaded.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixture: embedder with mock model
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder_with_mock(tmp_path):
    from services.indexing.embedder import CodeRankEmbedder
    e = CodeRankEmbedder(
        model_name="test-model",
        device="cpu",
        batch_size=8,
        cache_dir=str(tmp_path / "model"),
        vector_cache_dir=str(tmp_path / "vectors"),
    )
    mm = mock.MagicMock()
    mm.encode.return_value = np.zeros((1, 768), dtype=np.float32)
    e._model = mm
    return e, mm


# ===========================================================================
# EmbeddingCache tests
# ===========================================================================


def test_embedding_cache_set_get_roundtrip(tmp_path):
    from services.indexing.embedder import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path))
    vec = [0.5, 0.3, 0.1]
    cache.set("abc123", vec)
    result = cache.get("abc123")
    assert result is not None
    assert len(result) == 3
    assert result[0] == pytest.approx(0.5, abs=1e-5)


def test_embedding_cache_has_returns_false_for_missing(tmp_path):
    from services.indexing.embedder import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path))
    assert cache.has("nonexistent") is False


def test_embedding_cache_has_returns_true_after_set(tmp_path):
    from services.indexing.embedder import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path))
    cache.set("myhash", [1.0, 2.0])
    assert cache.has("myhash") is True


def test_embedding_cache_invalidate_removes_entry(tmp_path):
    from services.indexing.embedder import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path))
    cache.set("todelete", [1.0])
    cache.invalidate("todelete")
    assert cache.has("todelete") is False


def test_embedding_cache_get_returns_none_for_missing(tmp_path):
    from services.indexing.embedder import EmbeddingCache
    cache = EmbeddingCache(str(tmp_path))
    assert cache.get("nope") is None


# ===========================================================================
# CodeRankEmbedder.embed_texts tests
# ===========================================================================


def test_embed_texts_returns_empty_for_empty_input(embedder_with_mock):
    embedder, mm = embedder_with_mock
    result = embedder.embed_texts([])
    assert result == []
    mm.encode.assert_not_called()


def test_embed_texts_without_hashes_embeds_all(embedder_with_mock):
    embedder, mm = embedder_with_mock
    mm.encode.return_value = np.zeros((3, 768), dtype=np.float32)
    result = embedder.embed_texts(["a", "b", "c"])
    assert len(result) == 3
    assert all(len(v) == 768 for v in result)


def test_embed_texts_uses_cache_for_known_hashes(tmp_path):
    from services.indexing.embedder import CodeRankEmbedder
    embedder = CodeRankEmbedder(
        model_name="test",
        device="cpu",
        batch_size=8,
        cache_dir=str(tmp_path / "model"),
        vector_cache_dir=str(tmp_path / "vectors"),
    )
    # Pre-populate the cache with hash1
    cached_vec = [1.0] * 768
    embedder._cache.set("hash1", cached_vec)

    mm = mock.MagicMock()
    mm.encode.return_value = np.zeros((1, 768), dtype=np.float32)
    embedder._model = mm

    embedder.embed_texts(["text1", "text2"], content_hashes=["hash1", "hash2"])
    # encode should only be called with the uncached text (text2)
    called_texts = mm.encode.call_args[0][0]
    assert len(called_texts) == 1  # only 1 uncached text


def test_embed_query_prepends_codererank_prefix(embedder_with_mock):
    embedder, mm = embedder_with_mock
    mm.encode.return_value = np.zeros((1, 768), dtype=np.float32)
    embedder.embed_query("find auth logic")
    called_texts = mm.encode.call_args[0][0]
    assert len(called_texts) == 1
    assert called_texts[0].startswith("Represent this code snippet for searching relevant passages: ")
    assert "find auth logic" in called_texts[0]


def test_embed_texts_hash_mismatch_falls_back_to_no_cache(embedder_with_mock):
    embedder, mm = embedder_with_mock
    mm.encode.return_value = np.zeros((2, 768), dtype=np.float32)
    # Provide 2 texts but only 1 hash — mismatch, should fall back to embedding all
    result = embedder.embed_texts(["a", "b"], content_hashes=["only_one_hash"])
    assert len(result) == 2
    # encode called with all 2 texts (no cache used)
    called_texts = mm.encode.call_args[0][0]
    assert len(called_texts) == 2
