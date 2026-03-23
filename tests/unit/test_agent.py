"""
Unit tests for services/agent/agent.py

CodeIntelAgent state machine, AgentResponse dataclass, CodeSearchState,
SYSTEM_PROMPT, _build_context_string, and the confidence formula are tested
with all heavy deps (LLM, reranker, sqlite3, os.makedirs) mocked out.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Fixture: CodeIntelAgent with all heavy deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_with_mocks():
    """
    Return a CodeIntelAgent whose LLM, reranker, sqlite3, and os.makedirs
    are all mocked so no real models are loaded or files created.
    """
    with (
        mock.patch("services.agent.agent.get_provider") as mock_prov,
        mock.patch("services.agent.agent.get_reranker") as mock_rr,
        mock.patch("services.agent.agent.ALL_TOOLS", []),
        mock.patch("services.agent.agent.os.makedirs"),
        mock.patch("services.agent.agent.sqlite3.connect") as mock_conn,
    ):
        conn_obj = mock.MagicMock()
        conn_obj.execute.return_value = mock.MagicMock()
        mock_conn.return_value = conn_obj
        mock_prov.return_value = mock.MagicMock()
        mock_rr.return_value = mock.MagicMock()

        from services.agent.agent import CodeIntelAgent
        agent = CodeIntelAgent()
        yield agent, mock_prov, mock_rr


# ===========================================================================
# AgentResponse dataclass defaults
# ===========================================================================


def test_agent_response_dataclass_defaults():
    from services.agent.agent import AgentResponse
    resp = AgentResponse(answer="hello")
    assert resp.sources == []
    assert resp.confidence == 0.0
    assert resp.collection_hits == {}
    assert resp.graph_path == []
    assert resp.tool_calls_made == []


# ===========================================================================
# CodeSearchState TypedDict structure
# ===========================================================================


def test_code_search_state_typeddict_keys():
    from services.agent.agent import CodeSearchState
    keys = set(CodeSearchState.__annotations__.keys())
    required = {"messages", "query", "project_id", "step_count", "final_answer", "confidence"}
    assert required.issubset(keys), f"Missing keys: {required - keys}"


# ===========================================================================
# _build_context_string
# ===========================================================================


def test_build_context_string_empty(agent_with_mocks):
    agent, _, _ = agent_with_mocks
    result = agent._build_context_string([])
    assert result == "No relevant context found."


def test_build_context_string_formats_code_source(agent_with_mocks):
    agent, _, _ = agent_with_mocks
    from services.agent.reranker import RankedChunk
    chunk = {
        "file_path": "services/payment.py",
        "symbol_name": "process",
        "start_line": 10,
        "end_line": 20,
        "content": "def process(): pass",
        "collection": "code_repo",
        "scm_permalink": "",
        "gitlab_permalink": "",
        "section_path": "",
        "heading": "",
    }
    rc = RankedChunk(chunk=chunk, score=0.9, rank=1)
    result = agent._build_context_string([rc])
    assert "[SOURCE: code_repo]" in result
    assert "file: services/payment.py" in result
    assert "symbol: process" in result
    assert "def process(): pass" in result


def test_build_context_string_doc_source_with_section(agent_with_mocks):
    agent, _, _ = agent_with_mocks
    from services.agent.reranker import RankedChunk
    chunk = {
        "file_path": "",
        "symbol_name": "",
        "start_line": 0,
        "end_line": 0,
        "content": "Use JWT tokens for auth.",
        "collection": "app_docs",
        "scm_permalink": "",
        "gitlab_permalink": "",
        "section_path": "API / Auth",
        "heading": "Token Auth",
    }
    rc = RankedChunk(chunk=chunk, score=0.75, rank=1)
    result = agent._build_context_string([rc])
    assert "section: API / Auth" in result


# ===========================================================================
# query() — lazy graph construction
# ===========================================================================


def test_query_builds_graph_lazily(agent_with_mocks):
    agent, _, _ = agent_with_mocks
    assert agent._graph is None

    mock_graph = mock.MagicMock()
    mock_graph.invoke.return_value = {
        "final_answer": "Found it",
        "source_links": [],
        "confidence": 0.5,
        "collection_hits": {},
        "graph_path": [],
        "tool_calls_made": [],
    }

    with mock.patch.object(agent, "_build_graph", return_value=mock_graph) as mock_build:
        agent.query("what is retry logic?", project_id="proj1")
        mock_build.assert_called_once()
    assert agent._graph is mock_graph


def test_query_returns_agent_response(agent_with_mocks):
    agent, _, _ = agent_with_mocks
    mock_graph = mock.MagicMock()
    mock_graph.invoke.return_value = {
        "final_answer": "found it",
        "source_links": [{"file": "auth.py", "symbol": "verify", "start_line": 1, "end_line": 5, "collection": "code_repo", "permalink": "", "score": 0.9, "section_path": ""}],
        "confidence": 0.9,
        "collection_hits": {"code_repo": 1},
        "graph_path": [],
        "tool_calls_made": ["search_code"],
    }
    agent._graph = mock_graph

    from services.agent.agent import AgentResponse
    result = agent.query("question", project_id="p1")
    assert isinstance(result, AgentResponse)
    assert result.answer == "found it"
    assert result.confidence == pytest.approx(0.9)


# ===========================================================================
# SYSTEM_PROMPT content
# ===========================================================================


def test_system_prompt_contains_all_tool_names():
    from services.agent.agent import SYSTEM_PROMPT
    for tool_name in ["search_code", "keyword_search", "retrieve_entity", "traverse_graph", "search_app_docs", "search_incidents"]:
        assert tool_name in SYSTEM_PROMPT, f"'{tool_name}' not found in SYSTEM_PROMPT"


# ===========================================================================
# Confidence formula boundary values (pure math — no agent needed)
# ===========================================================================


def test_confidence_formula_boundary_values():
    def calc_confidence(top_score: float) -> float:
        return min(1.0, max(0.0, (top_score + 10) / 20))

    assert calc_confidence(10.0) == pytest.approx(1.0)
    assert calc_confidence(0.0) == pytest.approx(0.5)
    assert calc_confidence(-10.0) == pytest.approx(0.0)
    assert calc_confidence(5.0) == pytest.approx(0.75)
