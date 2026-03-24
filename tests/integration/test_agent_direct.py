"""
Integration-style tests for services/agent/agent.py

Tests AgentSettings env-var wiring, CodeIntelAgent initialization,
get_agent() singleton factory, lazy graph construction, graph reuse,
and the full query() pipeline response shape.

All heavy dependencies (LLM, reranker, SQLite, os.makedirs) are mocked
so these tests run fully offline without Docker or Ollama.

Converted from scripts/test_agent_direct.py into a repeatable pytest suite.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: CodeIntelAgent with all heavy deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_fixture():
    """
    Yield a CodeIntelAgent with LLM, reranker, sqlite3, and os.makedirs
    fully mocked — no real models loaded, no files created.
    """
    with (
        mock.patch("services.agent.agent.get_provider") as mock_prov,
        mock.patch("services.agent.agent.get_reranker") as mock_rr,
        mock.patch("services.agent.agent.ALL_TOOLS", []),
        mock.patch("services.agent.agent.os.makedirs"),
        mock.patch("services.agent.agent.sqlite3.connect") as mock_conn,
    ):
        mock_conn.return_value = mock.MagicMock()
        mock_prov.return_value = mock.MagicMock()
        mock_rr.return_value = mock.MagicMock()

        from services.agent.agent import CodeIntelAgent
        yield CodeIntelAgent(), mock_prov, mock_rr


# ---------------------------------------------------------------------------
# Helper: build a minimal final_state dict for query() to consume
# ---------------------------------------------------------------------------


def _make_final_state(
    answer="Test answer",
    confidence=0.85,
    sources=None,
    graph_path=None,
    collection_hits=None,
    tool_calls=None,
):
    return {
        "final_answer": answer,
        "confidence": confidence,
        "source_links": sources or [
            {
                "file": "services/main.py",
                "symbol": "main",
                "start_line": 1,
                "end_line": 10,
                "collection": "code_repo",
                "permalink": "",
                "score": 0.9,
                "section_path": "",
            }
        ],
        "reranked_chunks": [],
        "graph_path": graph_path or ["services/main.py"],
        "collection_hits": collection_hits or {"code_repo": 1, "app_docs": 0, "incident_reports": 0},
        "tool_calls_made": tool_calls or ["search_code"],
    }


# ===========================================================================
# AgentSettings — env var wiring
# ===========================================================================


def test_agent_settings_max_steps_field_default_is_10():
    """Check the class-level field default (not the .env-loaded runtime value)."""
    from services.agent.agent import AgentSettings
    assert AgentSettings.model_fields["agent_max_steps"].default == 10


def test_agent_settings_max_steps_is_positive_integer():
    from services.agent.agent import AgentSettings
    assert AgentSettings().agent_max_steps > 0


def test_agent_settings_rerank_top_k_default():
    from services.agent.agent import AgentSettings
    assert AgentSettings().agent_rerank_top_k == 5


def test_agent_settings_checkpoint_db_contains_agent_checkpoints():
    from services.agent.agent import AgentSettings
    assert "agent_checkpoints" in AgentSettings().agent_checkpoint_db


def test_agent_settings_reads_max_steps_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "25")
    from services.agent.agent import AgentSettings
    assert AgentSettings().agent_max_steps == 25


def test_agent_settings_reads_rerank_top_k_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_RERANK_TOP_K", "12")
    from services.agent.agent import AgentSettings
    assert AgentSettings().agent_rerank_top_k == 12


# ===========================================================================
# CodeIntelAgent — initialization
# ===========================================================================


def test_agent_calls_get_provider_on_init(agent_fixture):
    _, mock_prov, _ = agent_fixture
    mock_prov.assert_called_once()


def test_agent_calls_get_reranker_on_init(agent_fixture):
    _, _, mock_rr = agent_fixture
    mock_rr.assert_called_once()


def test_agent_graph_is_none_before_any_query(agent_fixture):
    agent, _, _ = agent_fixture
    assert agent._graph is None


def test_agent_has_cfg_attribute(agent_fixture):
    agent, _, _ = agent_fixture
    from services.agent.agent import AgentSettings
    assert isinstance(agent._cfg, AgentSettings)


def test_agent_llm_provider_is_set(agent_fixture):
    agent, mock_prov, _ = agent_fixture
    assert agent._llm_provider is mock_prov.return_value


def test_agent_reranker_is_set(agent_fixture):
    agent, _, mock_rr = agent_fixture
    assert agent._reranker is mock_rr.return_value


# ===========================================================================
# Lazy graph construction
# ===========================================================================


def test_agent_builds_graph_on_first_query(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state()

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph) as mock_build:
        agent.query("What is the entry point?", project_id="test/repo")
        mock_build.assert_called_once()
    assert agent._graph is fake_graph


def test_agent_does_not_rebuild_graph_on_second_query(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state()

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph) as mock_build:
        agent.query("q1", project_id="p")
        agent.query("q2", project_id="p")
        assert mock_build.call_count == 1


# ===========================================================================
# get_agent() — singleton factory
# ===========================================================================


def test_get_agent_returns_same_instance():
    with (
        mock.patch("services.agent.agent.get_provider"),
        mock.patch("services.agent.agent.get_reranker"),
        mock.patch("services.agent.agent.ALL_TOOLS", []),
        mock.patch("services.agent.agent.os.makedirs"),
        mock.patch("services.agent.agent.sqlite3.connect"),
    ):
        import services.agent.agent as agent_mod

        original = agent_mod._agent_instance
        agent_mod._agent_instance = None
        try:
            a1 = agent_mod.get_agent()
            a2 = agent_mod.get_agent()
            assert a1 is a2
        finally:
            agent_mod._agent_instance = original


def test_get_agent_returns_code_intel_agent_type():
    with (
        mock.patch("services.agent.agent.get_provider"),
        mock.patch("services.agent.agent.get_reranker"),
        mock.patch("services.agent.agent.ALL_TOOLS", []),
        mock.patch("services.agent.agent.os.makedirs"),
        mock.patch("services.agent.agent.sqlite3.connect"),
    ):
        import services.agent.agent as agent_mod
        from services.agent.agent import CodeIntelAgent

        original = agent_mod._agent_instance
        agent_mod._agent_instance = None
        try:
            a = agent_mod.get_agent()
            assert isinstance(a, CodeIntelAgent)
        finally:
            agent_mod._agent_instance = original


# ===========================================================================
# query() — AgentResponse shape
# ===========================================================================


def test_query_returns_agent_response_type(agent_fixture):
    from services.agent.agent import AgentResponse
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state()

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        result = agent.query("What is the main entry point?", project_id="test/repo")

    assert isinstance(result, AgentResponse)


def test_query_answer_matches_final_answer(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state(answer="Entry point is main.py line 1")

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        result = agent.query("entry point?", project_id="test/repo")

    assert result.answer == "Entry point is main.py line 1"


def test_query_confidence_matches_state(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state(confidence=0.92)

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        result = agent.query("confidence test", project_id="test/repo")

    assert result.confidence == 0.92


def test_query_graph_path_matches_state(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state(
        graph_path=["services/agent/agent.py", "services/llm/factory.py"]
    )

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        result = agent.query("graph path test", project_id="test/repo")

    assert result.graph_path == ["services/agent/agent.py", "services/llm/factory.py"]


def test_query_tool_calls_made_matches_state(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state(
        tool_calls=["search_code", "keyword_search", "retrieve_entity"]
    )

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        result = agent.query("tool calls test", project_id="test/repo")

    assert result.tool_calls_made == ["search_code", "keyword_search", "retrieve_entity"]


def test_query_collection_hits_matches_state(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    hits = {"code_repo": 5, "app_docs": 2, "incident_reports": 0}
    fake_graph.invoke.return_value = _make_final_state(collection_hits=hits)

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        result = agent.query("collection hits test", project_id="test/repo")

    assert result.collection_hits == hits


def test_query_answer_is_string_when_final_answer_empty(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.return_value = _make_final_state(answer="", confidence=0.0, sources=[], tool_calls=[])

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        result = agent.query("empty answer", project_id="test/repo")

    assert isinstance(result.answer, str)


def test_query_propagates_exception_from_graph(agent_fixture):
    agent, _, _ = agent_fixture
    fake_graph = mock.MagicMock()
    fake_graph.invoke.side_effect = RuntimeError("graph exploded")

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        with pytest.raises(RuntimeError, match="graph exploded"):
            agent.query("boom", project_id="test/repo")


def test_query_passes_project_id_in_initial_message(agent_fixture):
    """Verify the HumanMessage includes the project_id so the LLM is scoped."""
    agent, _, _ = agent_fixture
    captured_states = []

    def capture_invoke(state, config=None):
        captured_states.append(state)
        return _make_final_state()

    fake_graph = mock.MagicMock()
    fake_graph.invoke.side_effect = capture_invoke

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        agent.query("What does foo do?", project_id="myorg/myrepo")

    assert len(captured_states) == 1
    state = captured_states[0]
    assert state["project_id"] == "myorg/myrepo"
    # The HumanMessage content must mention the project_id
    human_msg = state["messages"][-1]
    assert "myorg/myrepo" in human_msg.content


def test_query_passes_question_in_state(agent_fixture):
    agent, _, _ = agent_fixture
    captured_states = []

    def capture_invoke(state, config=None):
        captured_states.append(state)
        return _make_final_state()

    fake_graph = mock.MagicMock()
    fake_graph.invoke.side_effect = capture_invoke

    with mock.patch.object(agent, "_build_graph", return_value=fake_graph):
        agent.query("How does retry logic work?", project_id="p")

    state = captured_states[0]
    assert state["query"] == "How does retry logic work?"
