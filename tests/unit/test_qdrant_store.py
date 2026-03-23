"""
Unit tests for services/indexing/qdrant_store.py

QdrantClient is patched so no real Qdrant server is needed.
Tests cover: filter building, hybrid search RRF fusion, error handling,
batch upsert sizing, delete, and collection health checks.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Fixture: QdrantStore with mocked client
# ---------------------------------------------------------------------------


@pytest.fixture
def store_and_client():
    with mock.patch("services.indexing.qdrant_store.QdrantClient") as MockClient:
        client = mock.MagicMock()
        MockClient.return_value = client
        from services.indexing.qdrant_store import QdrantStore
        store = QdrantStore(url="http://fake:6333")
        yield store, client


def _make_qdrant_hit(point_id: str, payload: dict):
    hit = mock.MagicMock()
    hit.id = point_id
    hit.payload = payload
    return hit


# ===========================================================================
# _build_filter tests
# ===========================================================================


def test_build_filter_single_keyword(store_and_client):
    from qdrant_client.http import models as qmodels
    store, _ = store_and_client
    f = store._build_filter({"language": "python"})
    assert isinstance(f, qmodels.Filter)
    assert len(f.must) == 1
    cond = f.must[0]
    assert cond.key == "language"
    assert isinstance(cond.match, qmodels.MatchValue)
    assert cond.match.value == "python"


def test_build_filter_list_value_uses_match_any(store_and_client):
    from qdrant_client.http import models as qmodels
    store, _ = store_and_client
    f = store._build_filter({"language": ["python", "go"]})
    cond = f.must[0]
    assert isinstance(cond.match, qmodels.MatchAny)
    assert set(cond.match.any) == {"python", "go"}


# ===========================================================================
# hybrid_search tests
# ===========================================================================


def test_hybrid_search_rrf_fused_results(store_and_client):
    store, client = store_and_client
    # dense search returns id1, id2, id3
    dense_hits = [
        _make_qdrant_hit("id1", {"content": "a"}),
        _make_qdrant_hit("id2", {"content": "b"}),
        _make_qdrant_hit("id3", {"content": "c"}),
    ]
    # text search returns id3 first (rank 1), id4 second — id3 is in both
    text_hits = [
        _make_qdrant_hit("id3", {"content": "c"}),
        _make_qdrant_hit("id4", {"content": "d"}),
    ]
    client.search.return_value = dense_hits
    client.scroll.return_value = (text_hits, None)

    results = store.hybrid_search("code_repo", [0.1] * 768, "auth", top_k=5)
    ids = [r["chunk_id"] for r in results]
    # id3 appears in both lists so should rank first
    assert ids[0] == "id3"
    assert set(ids) == {"id1", "id2", "id3", "id4"}


def test_hybrid_search_returns_empty_on_collection_not_found(store_and_client):
    from qdrant_client.http.exceptions import UnexpectedResponse
    store, client = store_and_client
    exc = UnexpectedResponse(status_code=404, reason_phrase=b"Not found", content=b"Not found", headers={})
    client.search.side_effect = exc
    result = store.hybrid_search("missing_collection", [0.0] * 768, "x")
    assert result == []


def test_hybrid_search_text_failure_is_gracefully_skipped(store_and_client):
    store, client = store_and_client
    dense_hits = [
        _make_qdrant_hit("id1", {"content": "a"}),
        _make_qdrant_hit("id2", {"content": "b"}),
    ]
    client.search.return_value = dense_hits
    # Text search (scroll) fails
    client.scroll.side_effect = Exception("index not ready")

    results = store.hybrid_search("code_repo", [0.1] * 768, "auth", top_k=5)
    # Should still return the 2 dense results
    assert len(results) == 2
    ids = {r["chunk_id"] for r in results}
    assert ids == {"id1", "id2"}


# ===========================================================================
# upsert_chunks tests
# ===========================================================================


def test_upsert_chunks_batches_correctly(store_and_client):
    from services.indexing.qdrant_store import BATCH_UPSERT_SIZE
    store, client = store_and_client

    # Build 250 mock Chunk objects (BATCH_UPSERT_SIZE=100 → 3 batches: 100+100+50)
    from services.parsing.chunker import Chunk
    import hashlib
    chunks = []
    for i in range(250):
        content = f"content {i}"
        cid = hashlib.sha256(content.encode()).hexdigest()
        chunks.append(Chunk(id=cid, content=content, metadata={"file_path": f"file{i}.py"}))
    vectors = [[0.1] * 768] * 250

    store.upsert_chunks("code_repo", chunks, vectors)
    assert client.upsert.call_count == 3  # ceil(250/100) == 3


def test_upsert_chunks_raises_on_length_mismatch(store_and_client):
    store, _ = store_and_client
    from services.parsing.chunker import Chunk
    import hashlib
    chunks = [Chunk(id=hashlib.sha256(b"x").hexdigest(), content="x", metadata={})] * 3
    vectors = [[0.1] * 768] * 2  # mismatch
    with pytest.raises(ValueError, match="Mismatch"):
        store.upsert_chunks("code_repo", chunks, vectors)


# ===========================================================================
# delete_by_file tests
# ===========================================================================


def test_delete_by_file_returns_1_on_completed(store_and_client):
    from qdrant_client.http.models import UpdateStatus
    store, client = store_and_client
    result_obj = mock.MagicMock()
    result_obj.status = UpdateStatus.COMPLETED
    client.delete.return_value = result_obj
    count = store.delete_by_file("code_repo", "auth.py", "proj1")
    assert count == 1


# ===========================================================================
# is_collection_populated tests
# ===========================================================================


def test_is_collection_populated_returns_false_on_exception(store_and_client):
    store, client = store_and_client
    client.get_collection.side_effect = Exception("not found")
    assert store.is_collection_populated("code_repo") is False


def test_is_collection_populated_returns_false_when_zero_points(store_and_client):
    store, client = store_and_client
    info = mock.MagicMock()
    info.points_count = 0
    client.get_collection.return_value = info
    assert store.is_collection_populated("code_repo") is False
