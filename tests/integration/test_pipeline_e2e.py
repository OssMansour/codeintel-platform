"""
End-to-end pipeline tests: Parse → Chunk → Embed → Store → Retrieve

Exercises the full ingestion pipeline offline:

    parse_file()          → FileSymbolTable
    chunk_file()          → list[Chunk]          (real code)
    embedder.embed_texts()→ list[list[float]]     (deterministic mock)
    store.upsert_chunks() → persisted in Qdrant   (qdrant-client :memory:)
    store.hybrid_search() → retrieved results     (real RRF logic)

No Docker, no network, no model weights required.
"""
from __future__ import annotations

import sys
import os
import unittest.mock as mock
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING level — quiet tests
    logger_factory=structlog.PrintLoggerFactory(),
)

# ---------------------------------------------------------------------------
# Shared sample source  (realistic enough to exercise all symbol types)
# ---------------------------------------------------------------------------

_SAMPLE_SOURCE = '''\
"""Module for testing the ingestion pipeline end-to-end."""

import os
import sys


class PaymentService:
    """Handles payment processing operations."""

    def process_payment(self, amount: float, user_id: str) -> bool:
        """Process a payment transaction for the given user."""
        if amount <= 0:
            return False
        return True

    def refund_payment(self, transaction_id: str) -> bool:
        """Refund a previously processed payment by transaction ID."""
        return False


def calculate_retry_backoff(attempt: int, base_delay: float = 1.0) -> float:
    """Calculate exponential backoff delay for retry logic."""
    return base_delay * (2 ** attempt)


def health_check() -> dict:
    """Return service health status dictionary."""
    return {"status": "ok", "version": "1.0"}
'''

PROJECT_ID = "test-org/test-repo"
BRANCH = "main"
COMMIT = "abc123def456"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_py_file(tmp_path_factory):
    """Write the sample Python source to a temp .py file."""
    d = tmp_path_factory.mktemp("src")
    f = d / "payment_service.py"
    f.write_text(_SAMPLE_SOURCE, encoding="utf-8")
    return str(f)


@pytest.fixture(scope="module")
def parsed_table(sample_py_file):
    """Return the FileSymbolTable for the sample file."""
    from services.parsing.parser import parse_file
    return parse_file(sample_py_file)


@pytest.fixture(scope="module")
def chunks(parsed_table, sample_py_file):
    """Return chunks produced by chunk_file() for the sample file."""
    from services.parsing.chunker import chunk_file
    return chunk_file(
        symbol_table=parsed_table,
        project_id=PROJECT_ID,
        repo_url="https://github.com/test-org/test-repo",
        branch=BRANCH,
        commit_sha=COMMIT,
        scm_provider="github",
        scm_base_url="https://github.com",
    )


@pytest.fixture(scope="module")
def mock_embedder():
    """
    Deterministic mock embedder.

    Each text gets a unique but reproducible 768-dim vector based on the
    SHA-256 of its content — close enough for RRF ranking tests.
    """
    def _vec_for(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Expand 32 bytes into 768 floats in [0, 1]
        base = [b / 255.0 for b in digest]
        return (base * 25)[:768]

    embedder = mock.MagicMock()
    embedder.dimension = 768
    embedder.embed_texts.side_effect = (
        lambda texts, content_hashes=None, use_query_prefix=False:
        [_vec_for(t) for t in texts]
    )
    embedder.embed_query.side_effect = _vec_for
    return embedder


@pytest.fixture(scope="module")
def in_memory_store():
    """
    QdrantStore backed by an in-memory Qdrant client.

    Bypasses network entirely — no Docker needed.
    """
    from qdrant_client import QdrantClient
    from services.indexing.qdrant_store import QdrantStore, QdrantSettings

    store = object.__new__(QdrantStore)
    store._client = QdrantClient(":memory:")
    store._collection_code = "code_repo"
    store._collection_app_docs = "app_docs"
    store._collection_incidents = "incident_reports"
    store._cfg = QdrantSettings()
    store._url = ":memory:"
    store._log = structlog.get_logger("test.store")

    # Bootstrap collections (idempotent)
    store.setup_collections()
    return store


@pytest.fixture(scope="module")
def populated_store(in_memory_store, chunks, mock_embedder):
    """
    Store with all sample chunks already upserted.
    Returns (store, chunks, vectors) for downstream retrieval tests.
    """
    contents = [c.content for c in chunks]
    vectors = mock_embedder.embed_texts(contents)
    in_memory_store.upsert_chunks("code_repo", chunks, vectors)
    return in_memory_store, chunks, vectors


# ===========================================================================
# Stage 1 — Parse
# ===========================================================================


def test_parse_returns_file_symbol_table(parsed_table):
    from services.parsing.parser import FileSymbolTable
    assert isinstance(parsed_table, FileSymbolTable)


def test_parse_detects_python(parsed_table):
    assert parsed_table.language == "python"


def test_parse_extracts_payment_service_class(parsed_table):
    class_names = [c.name for c in parsed_table.classes]
    assert "PaymentService" in class_names


def test_parse_extracts_standalone_functions(parsed_table):
    func_names = [f.name for f in parsed_table.functions]
    assert "calculate_retry_backoff" in func_names
    assert "health_check" in func_names


def test_parse_extracts_methods_inside_class(parsed_table):
    func_names = [f.name for f in parsed_table.functions]
    assert "process_payment" in func_names
    assert "refund_payment" in func_names


def test_parse_extracts_module_docstring(parsed_table):
    assert "ingestion pipeline" in parsed_table.module_docstring.lower()


def test_parse_extracts_imports(parsed_table):
    assert len(parsed_table.imports) >= 2


def test_parse_source_lines_match_file(parsed_table):
    expected = len(_SAMPLE_SOURCE.splitlines(keepends=True))
    assert len(parsed_table.source_lines) == expected


def test_parse_all_symbols_sorted_by_line(parsed_table):
    syms = parsed_table.all_symbols()
    lines = [s.start_line for s in syms]
    assert lines == sorted(lines)


# ===========================================================================
# Stage 2 — Chunk
# ===========================================================================


def test_chunk_produces_at_least_four_chunks(chunks):
    # class + 2 methods + 2 functions = at least 4
    assert len(chunks) >= 4


def test_chunk_ids_are_64_hex_chars(chunks):
    for c in chunks:
        assert len(c.id) == 64
        assert all(ch in "0123456789abcdef" for ch in c.id)


def test_chunk_ids_are_unique(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


def test_chunk_metadata_has_required_fields(chunks):
    required = {
        "project_id", "file_path", "language", "symbol_name",
        "symbol_type", "start_line", "end_line", "content_hash",
        "source_collection", "trust_level", "last_indexed_at",
    }
    for c in chunks:
        missing = required - set(c.metadata.keys())
        assert not missing, f"Chunk {c.id[:8]} missing fields: {missing}"


def test_chunk_metadata_project_id_matches(chunks):
    for c in chunks:
        assert c.metadata["project_id"] == PROJECT_ID


def test_chunk_metadata_language_is_python(chunks):
    for c in chunks:
        assert c.metadata["language"] == "python"


def test_chunk_metadata_source_collection_is_code_repo(chunks):
    for c in chunks:
        assert c.metadata["source_collection"] == "code_repo"


def test_chunk_metadata_trust_level_is_code(chunks):
    for c in chunks:
        assert c.metadata["trust_level"] == "code"


def test_chunk_for_process_payment_has_correct_symbol_type(chunks):
    method_chunks = [c for c in chunks if c.metadata["symbol_name"] == "process_payment"]
    assert method_chunks, "process_payment chunk not found"
    assert method_chunks[0].metadata["symbol_type"] == "method"


def test_chunk_for_calculate_retry_backoff_is_function(chunks):
    fn_chunks = [c for c in chunks if c.metadata["symbol_name"] == "calculate_retry_backoff"]
    assert fn_chunks, "calculate_retry_backoff chunk not found"
    assert fn_chunks[0].metadata["symbol_type"] == "function"


def test_chunk_content_includes_docstring(chunks):
    """Chunker prepends docstrings for richer embedding semantics."""
    fn_chunks = [c for c in chunks if c.metadata["symbol_name"] == "calculate_retry_backoff"]
    assert fn_chunks, "calculate_retry_backoff chunk not found"
    assert "backoff" in fn_chunks[0].content.lower()


def test_chunk_ids_are_deterministic(sample_py_file):
    """Same file twice must produce identical chunk IDs."""
    from services.parsing.parser import parse_file
    from services.parsing.chunker import chunk_file

    t1 = parse_file(sample_py_file)
    t2 = parse_file(sample_py_file)
    c1 = chunk_file(t1, project_id=PROJECT_ID, branch=BRANCH, commit_sha=COMMIT)
    c2 = chunk_file(t2, project_id=PROJECT_ID, branch=BRANCH, commit_sha=COMMIT)
    assert [c.id for c in c1] == [c.id for c in c2]


def test_chunk_scm_permalink_contains_github_url(chunks):
    for c in chunks:
        permalink = c.metadata.get("scm_permalink", "")
        if permalink:
            assert "github.com" in permalink


def test_chunk_line_numbers_are_positive(chunks):
    for c in chunks:
        assert c.metadata["start_line"] >= 1
        assert c.metadata["end_line"] >= c.metadata["start_line"]


# ===========================================================================
# Stage 3 — Embed
# ===========================================================================


def test_embed_texts_called_with_chunk_contents(chunks, mock_embedder):
    contents = [c.content for c in chunks]
    vectors = mock_embedder.embed_texts(contents)
    mock_embedder.embed_texts.assert_called()
    assert len(vectors) == len(chunks)


def test_embed_vectors_have_correct_dimension(chunks, mock_embedder):
    contents = [c.content for c in chunks]
    vectors = mock_embedder.embed_texts(contents)
    for vec in vectors:
        assert len(vec) == 768


def test_embed_query_returns_768_dim_vector(mock_embedder):
    vec = mock_embedder.embed_query("find payment retry logic")
    assert len(vec) == 768


def test_embed_different_contents_produce_different_vectors(chunks, mock_embedder):
    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks")
    contents = [c.content for c in chunks[:2]]
    v1, v2 = mock_embedder.embed_texts(contents)
    assert v1 != v2, "Different content must produce different vectors"


# ===========================================================================
# Stage 4 — Store  (upsert + collection state)
# ===========================================================================


def test_store_setup_creates_code_repo_collection(in_memory_store):
    stats = in_memory_store.get_collection_stats("code_repo")
    assert stats["exists"] is True


def test_store_setup_creates_app_docs_collection(in_memory_store):
    stats = in_memory_store.get_collection_stats("app_docs")
    assert stats["exists"] is True


def test_store_setup_creates_incident_reports_collection(in_memory_store):
    stats = in_memory_store.get_collection_stats("incident_reports")
    assert stats["exists"] is True


def test_upsert_increases_point_count(populated_store):
    store, chunks, _ = populated_store
    stats = store.get_collection_stats("code_repo")
    assert stats["points_count"] >= len(chunks)


def test_collection_is_populated_after_upsert(populated_store):
    store, _, _ = populated_store
    assert store.is_collection_populated("code_repo") is True


def test_upsert_idempotent_same_chunks(populated_store, mock_embedder):
    """Upserting the same chunks twice must not increase point count."""
    store, chunks, vectors = populated_store
    before = store.get_collection_stats("code_repo")["points_count"]
    store.upsert_chunks("code_repo", chunks, vectors)
    after = store.get_collection_stats("code_repo")["points_count"]
    assert after == before


def test_get_all_collection_stats_returns_three_keys(populated_store):
    store, _, _ = populated_store
    stats = store.get_all_collection_stats()
    assert set(stats.keys()) == {"code_repo", "app_docs", "incident_reports"}


# ===========================================================================
# Stage 5 — Retrieve
# ===========================================================================


def test_hybrid_search_returns_results(populated_store, mock_embedder):
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("process payment transaction")
    results = store.hybrid_search("code_repo", q_vec, "process payment", top_k=10)
    assert len(results) > 0


def test_hybrid_search_results_have_required_fields(populated_store, mock_embedder):
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("payment service")
    results = store.hybrid_search("code_repo", q_vec, "payment", top_k=5)
    for r in results:
        assert "chunk_id" in r
        assert "score" in r
        assert "metadata" in r
        assert "content" in r


def test_hybrid_search_result_metadata_has_file_path(populated_store, mock_embedder):
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("payment service")
    results = store.hybrid_search("code_repo", q_vec, "payment", top_k=5)
    for r in results:
        assert "file_path" in r["metadata"]


def test_hybrid_search_result_metadata_has_symbol_name(populated_store, mock_embedder):
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("payment service")
    results = store.hybrid_search("code_repo", q_vec, "payment", top_k=5)
    for r in results:
        assert "symbol_name" in r["metadata"]


def test_hybrid_search_scores_are_positive(populated_store, mock_embedder):
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("retry backoff")
    results = store.hybrid_search("code_repo", q_vec, "retry", top_k=5)
    for r in results:
        assert r["score"] > 0.0


def test_hybrid_search_filter_by_language(populated_store, mock_embedder):
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("payment")
    results = store.hybrid_search(
        "code_repo", q_vec, "payment",
        filters={"language": "python"},
        top_k=10,
    )
    for r in results:
        assert r["metadata"]["language"] == "python"


def test_hybrid_search_filter_by_project_id(populated_store, mock_embedder):
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("health")
    results = store.hybrid_search(
        "code_repo", q_vec, "health",
        filters={"project_id": PROJECT_ID},
        top_k=10,
    )
    for r in results:
        assert r["metadata"]["project_id"] == PROJECT_ID


def test_hybrid_search_wrong_project_returns_no_cross_project_results(populated_store, mock_embedder):
    """Filtering on a non-existent project_id must return 0 results."""
    store, _, _ = populated_store
    q_vec = mock_embedder.embed_query("payment")
    results = store.hybrid_search(
        "code_repo", q_vec, "payment",
        filters={"project_id": "nonexistent/project"},
        top_k=10,
    )
    assert results == []


def test_hybrid_search_missing_collection_returns_empty(in_memory_store, mock_embedder):
    """Searching a non-existent collection must return [] not raise."""
    q_vec = mock_embedder.embed_query("anything")
    results = in_memory_store.hybrid_search("does_not_exist", q_vec, "anything", top_k=5)
    assert results == []


def test_hybrid_search_empty_collection_returns_empty(in_memory_store, mock_embedder):
    """Searching the unpopulated app_docs collection must return []."""
    q_vec = mock_embedder.embed_query("payment")
    results = in_memory_store.hybrid_search("app_docs", q_vec, "payment", top_k=5)
    assert results == []


# ===========================================================================
# Stage 6 — Full pipeline in one test  (parse → chunk → embed → store → retrieve)
# ===========================================================================


def test_full_pipeline_parse_chunk_embed_store_retrieve(tmp_path):
    """
    Single smoke test exercising every stage of the ingestion pipeline.

    Writes a Python file, parses it, chunks it, embeds with a mock,
    upserts into in-memory Qdrant, then retrieves and validates results.
    """
    from qdrant_client import QdrantClient
    from services.parsing.parser import parse_file
    from services.parsing.chunker import chunk_file
    from services.indexing.qdrant_store import QdrantStore, QdrantSettings

    # --- Write source ---
    src = tmp_path / "engine.py"
    src.write_text(
        '"""Planning engine module."""\n\n'
        'def plan_steps(goal: str) -> list:\n'
        '    """Generate a step-by-step plan for the given goal."""\n'
        '    return [goal]\n\n'
        'def execute_plan(steps: list) -> bool:\n'
        '    """Execute each step in the plan sequentially."""\n'
        '    return True\n',
        encoding="utf-8",
    )

    # --- Stage 1: Parse ---
    table = parse_file(str(src))
    assert table.language == "python"
    func_names = [f.name for f in table.functions]
    assert "plan_steps" in func_names
    assert "execute_plan" in func_names

    # --- Stage 2: Chunk ---
    pipeline_project = "e2e/test-project"
    file_chunks = chunk_file(
        table,
        project_id=pipeline_project,
        branch="main",
        commit_sha="deadbeef",
    )
    assert len(file_chunks) >= 2
    for c in file_chunks:
        assert c.metadata["project_id"] == pipeline_project

    # --- Stage 3: Embed (deterministic mock) ---
    vectors = [[0.5] * 768 for _ in file_chunks]

    # --- Stage 4: Store ---
    store = object.__new__(QdrantStore)
    store._client = QdrantClient(":memory:")
    store._collection_code = "code_repo"
    store._collection_app_docs = "app_docs"
    store._collection_incidents = "incident_reports"
    store._cfg = QdrantSettings()
    store._url = ":memory:"
    store._log = structlog.get_logger("test.e2e")
    store.setup_collections()

    store.upsert_chunks("code_repo", file_chunks, vectors)
    assert store.is_collection_populated("code_repo")

    # --- Stage 5: Retrieve ---
    q_vec = [0.5] * 768
    results = store.hybrid_search("code_repo", q_vec, "plan steps goal", top_k=10)

    assert len(results) > 0
    # All results must be from our project
    for r in results:
        assert r["metadata"]["project_id"] == pipeline_project

    # Verify we can find both functions
    found_symbols = {r["metadata"]["symbol_name"] for r in results}
    assert "plan_steps" in found_symbols
    assert "execute_plan" in found_symbols
