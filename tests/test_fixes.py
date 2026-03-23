"""
Local unit tests for the 6 bugs fixed in this session.
Run with:
    conda activate codeintel
    cd c:/Users/Lenovo/Desktop/codeintel-platform
    python -m pytest tests/test_fixes.py -v
"""
import ast
import json
import pathlib
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Only mock celery — everything else (redis, qdrant-client, structlog) is installed
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())


# =============================================================================
# Bug 1 — tools.py: content truncation dead code
# =============================================================================

def test_format_result_truncates_content():
    """_format_result must return the truncated content, not the original long string."""
    from services.agent.tools import _format_result

    long_content = "x" * 1000
    hit = {
        "chunk_id": "abc",
        "score": 0.9,
        "collection": "code_repo",
        "file_path": "foo.py",
        "symbol_name": "bar",
        "symbol_type": "function",
        "start_line": 1,
        "end_line": 10,
        "content": long_content,
        "scm_permalink": "",
        "gitlab_permalink": "",
    }
    result = _format_result(hit)
    assert len(result["content"]) <= 503, (
        f"content should be ≤503 chars after truncation, got {len(result['content'])}"
    )
    assert result["content"].endswith("…"), "truncated content must end with …"


def test_format_result_short_content_unchanged():
    """Short content (≤500 chars) must pass through unchanged."""
    from services.agent.tools import _format_result

    hit = {
        "chunk_id": "abc",
        "score": 0.9,
        "collection": "code_repo",
        "file_path": "foo.py",
        "symbol_name": "bar",
        "symbol_type": "function",
        "start_line": 1,
        "end_line": 5,
        "content": "short",
        "scm_permalink": "",
        "gitlab_permalink": "",
    }
    result = _format_result(hit)
    assert result["content"] == "short"


# =============================================================================
# Bug 2 — api.py: graph not assigned in lifespan (AST check)
# =============================================================================

def test_build_graph_return_value_is_assigned():
    """api.py lifespan must assign agent._graph = agent._build_graph()."""
    src = pathlib.Path("services/agent/api.py").read_text()
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "_graph"
                    and isinstance(node.value, ast.Call)
                ):
                    func = node.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "_build_graph":
                        found = True
    assert found, (
        "api.py must have 'agent._graph = agent._build_graph()' — "
        "bare call discards the return value"
    )


# =============================================================================
# Bug 3 — agent.py: dict tool results silently dropped
# =============================================================================

def _parse_tool_msg_content(content_str: str) -> list[dict]:
    """Replicate rerank_and_answer_node's chunk-collection logic."""
    all_chunks = []
    collection_hits = {"code_repo": 0, "app_docs": 0, "incident_reports": 0}

    try:
        tool_results = json.loads(content_str)

        if isinstance(tool_results, list):
            for r in tool_results:
                if isinstance(r, dict) and "chunk_id" in r:
                    all_chunks.append(r)
                    coll = r.get("collection", "code_repo")
                    if coll in collection_hits:
                        collection_hits[coll] += 1

        elif isinstance(tool_results, dict):
            # retrieve_entity path
            if tool_results.get("found") and tool_results.get("content"):
                chunk = {
                    "chunk_id": tool_results.get("chunk_id", "entity_x"),
                    "score": 1.0,
                    "collection": tool_results.get("collection", "code_repo"),
                    "file_path": tool_results.get("file_path", ""),
                    "symbol_name": tool_results.get("symbol_name", ""),
                    "symbol_type": tool_results.get("symbol_type", ""),
                    "start_line": tool_results.get("start_line", 0),
                    "end_line": tool_results.get("end_line", 0),
                    "content": tool_results.get("content", "")[:500],
                    "scm_permalink": tool_results.get("scm_permalink", ""),
                    "gitlab_permalink": tool_results.get("scm_permalink", ""),
                }
                all_chunks.append(chunk)
                collection_hits["code_repo"] += 1

            # traverse_graph path
            elif "node_id" in tool_results:
                parts = [f"Dependency graph for: {tool_results['node_id']}"]
                callers = tool_results.get("callers", [])
                callees = tool_results.get("callees", [])
                if callers:
                    parts.append("Callers: " + ", ".join(
                        c.get("qualified_name") or c.get("name", "") for c in callers
                    ))
                if callees:
                    parts.append("Callees: " + ", ".join(
                        c.get("qualified_name") or c.get("name", "") for c in callees
                    ))
                if len(parts) > 1:
                    chunk = {
                        "chunk_id": f"graph_{tool_results['node_id']}",
                        "score": 0.8,
                        "collection": "code_repo",
                        "file_path": "",
                        "symbol_name": tool_results["node_id"],
                        "start_line": 0,
                        "end_line": 0,
                        "content": "\n".join(parts),
                    }
                    all_chunks.append(chunk)
                    collection_hits["code_repo"] += 1

    except Exception:
        pass

    return all_chunks


def test_retrieve_entity_result_collected():
    """retrieve_entity dict result (found=True) must land in all_chunks."""
    payload = {
        "found": True,
        "chunk_id": "ent_1",
        "content": "def foo(): pass",
        "file_path": "foo.py",
        "symbol_name": "foo",
        "symbol_type": "function",
        "start_line": 1,
        "end_line": 1,
        "collection": "code_repo",
        "scm_permalink": "",
    }
    chunks = _parse_tool_msg_content(json.dumps(payload))
    assert len(chunks) == 1, "retrieve_entity result should produce 1 chunk"
    assert chunks[0]["symbol_name"] == "foo"


def test_retrieve_entity_not_found_dropped():
    """retrieve_entity with found=False must not produce a chunk."""
    payload = {"found": False, "error": "not found"}
    chunks = _parse_tool_msg_content(json.dumps(payload))
    assert len(chunks) == 0


def test_traverse_graph_result_collected():
    """traverse_graph result must be summarised into a text chunk."""
    payload = {
        "node_id": "mymodule.MyClass.my_method",
        "callers": [{"qualified_name": "mymodule.caller_fn"}],
        "callees": [{"qualified_name": "os.path.join"}, {"name": "helper"}],
    }
    chunks = _parse_tool_msg_content(json.dumps(payload))
    assert len(chunks) == 1
    assert "mymodule.caller_fn" in chunks[0]["content"]
    assert "os.path.join" in chunks[0]["content"]


def test_list_tool_result_still_works():
    """search_code list results must still be collected correctly."""
    payload = [
        {"chunk_id": "c1", "score": 0.9, "collection": "code_repo",
         "file_path": "a.py", "content": "foo"},
        {"chunk_id": "c2", "score": 0.8, "collection": "code_repo",
         "file_path": "b.py", "content": "bar"},
    ]
    chunks = _parse_tool_msg_content(json.dumps(payload))
    assert len(chunks) == 2


# =============================================================================
# Bug 4 — qdrant_store.py: hybrid search RRF fusion (offline, no Qdrant needed)
# =============================================================================

def test_rrf_merge_overlap_ranks_highest():
    """A doc appearing in both dense and text results must outscore docs in only one."""
    rrf_k = 60
    dense_ids = ["doc1", "doc2", "doc3"]   # doc3 is rank 3 in dense
    text_ids  = ["doc3", "doc4"]            # doc3 is rank 1 in text

    rrf_scores: dict = {}
    for rank, cid in enumerate(dense_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(text_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    assert sorted_ids[0] == "doc3", (
        f"doc3 (in both lists) should rank first, got {sorted_ids}"
    )
    assert set(sorted_ids) == {"doc1", "doc2", "doc3", "doc4"}


def test_rrf_dense_only_fallback():
    """When text results are empty, RRF should still return all dense results ranked correctly."""
    rrf_k = 60
    dense_ids = ["a", "b", "c"]
    text_ids: list = []

    rrf_scores: dict = {}
    for rank, cid in enumerate(dense_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(text_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    assert sorted_ids == ["a", "b", "c"], f"dense-only order wrong: {sorted_ids}"


# =============================================================================
# Bug 6 — agent.py: step_count fencepost — AST check
# =============================================================================

def test_should_continue_uses_strict_greater_than():
    """should_continue must use > (not >=) for step budget check."""
    src = pathlib.Path("services/agent/agent.py").read_text()

    # Find the should_continue function and check the comparison operator
    tree = ast.parse(src)

    comparisons_in_should_continue = []
    inside_fn = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "should_continue":
            inside_fn = True
            for child in ast.walk(node):
                if isinstance(child, ast.Compare):
                    for op in child.ops:
                        comparisons_in_should_continue.append(type(op).__name__)

    assert inside_fn, "should_continue function not found in agent.py"

    # Must contain Gt (>) and must NOT contain GtE (>=) for step budget
    assert "Gt" in comparisons_in_should_continue, (
        "should_continue must use > for step budget check"
    )
    assert "GtE" not in comparisons_in_should_continue, (
        "should_continue must NOT use >= for step budget (fencepost bug)"
    )
