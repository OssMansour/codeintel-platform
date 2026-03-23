"""
Unit tests for services/agent/reranker.py

CrossEncoderReranker and RankedChunk are tested with a mock model injected
directly — no real sentence-transformers model is loaded.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Fixture: reranker with mock model pre-injected
# ---------------------------------------------------------------------------


@pytest.fixture
def reranker():
    from services.agent.reranker import CrossEncoderReranker
    r = CrossEncoderReranker(model_name="test-model", cache_dir="/tmp/test_reranker")
    r._model = mock.MagicMock()
    return r


def _make_chunk(file_path="auth.py", symbol_name="verify", content="def verify(): pass", score=0.5, **kwargs):
    chunk = {
        "chunk_id": kwargs.get("chunk_id", "c1"),
        "file_path": file_path,
        "symbol_name": symbol_name,
        "content": content,
        "score": score,
        "collection": kwargs.get("collection", "code_repo"),
        "start_line": kwargs.get("start_line", 1),
        "end_line": kwargs.get("end_line", 5),
        "scm_permalink": kwargs.get("scm_permalink", ""),
        "gitlab_permalink": kwargs.get("gitlab_permalink", ""),
    }
    return chunk


# ===========================================================================
# RankedChunk property tests
# ===========================================================================


def test_ranked_chunk_file_path_from_top_level():
    from services.agent.reranker import RankedChunk
    rc = RankedChunk(chunk={"file_path": "a/b.py"}, score=1.0, rank=1)
    assert rc.file_path == "a/b.py"


def test_ranked_chunk_file_path_from_metadata_fallback():
    from services.agent.reranker import RankedChunk
    rc = RankedChunk(chunk={"metadata": {"file_path": "c/d.py"}}, score=1.0, rank=1)
    assert rc.file_path == "c/d.py"


def test_ranked_chunk_scm_permalink_prefers_scm():
    from services.agent.reranker import RankedChunk
    rc = RankedChunk(
        chunk={
            "scm_permalink": "https://github.com/repo",
            "gitlab_permalink": "https://gitlab.com/repo",
        },
        score=1.0,
        rank=1,
    )
    assert rc.scm_permalink == "https://github.com/repo"


def test_ranked_chunk_collection_from_metadata_source_collection():
    from services.agent.reranker import RankedChunk
    rc = RankedChunk(
        chunk={"metadata": {"source_collection": "app_docs"}},
        score=0.5,
        rank=1,
    )
    assert rc.collection == "app_docs"


# ===========================================================================
# CrossEncoderReranker.rerank tests
# ===========================================================================


def test_rerank_empty_chunks_returns_empty(reranker):
    result = reranker.rerank(query="auth logic", chunks=[], top_k=5)
    assert result == []
    reranker._model.predict.assert_not_called()


def test_rerank_scores_and_returns_top_k(reranker):
    import numpy as np
    chunks = [_make_chunk(file_path=f"file{i}.py", chunk_id=f"c{i}") for i in range(5)]
    # Scores correspond positionally to chunks
    reranker._model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.8])
    result = reranker.rerank(query="auth", chunks=chunks, top_k=3)
    assert len(result) == 3
    # Best score (0.9) should be rank 1
    assert result[0].score == pytest.approx(0.9)
    assert result[0].rank == 1
    assert result[1].rank == 2
    assert result[2].rank == 3


def test_rerank_truncates_content_at_2048(reranker):
    import numpy as np
    long_content = "x" * 3000
    chunk = _make_chunk(content=long_content)
    reranker._model.predict.return_value = np.array([0.5])
    reranker.rerank(query="q", chunks=[chunk], top_k=1)
    pairs = reranker._model.predict.call_args[0][0]  # positional arg
    # The document part of the pair should be truncated
    _, doc = pairs[0]
    assert len(doc) <= 2048


def test_rerank_falls_back_on_predict_error(reranker):
    reranker._model.predict.side_effect = RuntimeError("CUDA OOM")
    chunks = [_make_chunk(chunk_id=f"c{i}", score=float(i)) for i in range(3)]
    result = reranker.rerank(query="q", chunks=chunks, top_k=2)
    # Should fall back to original order, capped at top_k=2
    assert len(result) == 2
    assert result[0].rank == 1
    assert result[1].rank == 2


# ===========================================================================
# CrossEncoderReranker.rerank_with_diversity tests
# ===========================================================================


def test_rerank_with_diversity_limits_per_file(reranker):
    import numpy as np
    # 4 chunks from auth.py, 2 from payment.py
    auth_chunks = [
        _make_chunk(file_path="auth.py", chunk_id=f"a{i}", score=0.9) for i in range(4)
    ]
    payment_chunks = [
        _make_chunk(file_path="payment.py", chunk_id=f"p{i}", score=0.5) for i in range(2)
    ]
    all_chunks = auth_chunks + payment_chunks
    # Auth chunks score higher — but diversity should cap them at max_per_file=2
    scores = [0.9, 0.88, 0.85, 0.82, 0.5, 0.48]
    reranker._model.predict.return_value = np.array(scores)
    result = reranker.rerank_with_diversity(query="auth", chunks=all_chunks, top_k=4, max_per_file=2)
    auth_count = sum(1 for rc in result if rc.file_path == "auth.py")
    payment_count = sum(1 for rc in result if rc.file_path == "payment.py")
    assert auth_count == 2
    assert payment_count == 2


def test_rerank_with_diversity_reassigns_contiguous_ranks(reranker):
    import numpy as np
    auth_chunks = [_make_chunk(file_path="auth.py", chunk_id=f"a{i}") for i in range(4)]
    payment_chunks = [_make_chunk(file_path="payment.py", chunk_id=f"p{i}") for i in range(2)]
    all_chunks = auth_chunks + payment_chunks
    scores = [0.9, 0.88, 0.85, 0.82, 0.5, 0.48]
    reranker._model.predict.return_value = np.array(scores)
    result = reranker.rerank_with_diversity(query="auth", chunks=all_chunks, top_k=4, max_per_file=2)
    ranks = [rc.rank for rc in result]
    assert ranks == [1, 2, 3, 4]
