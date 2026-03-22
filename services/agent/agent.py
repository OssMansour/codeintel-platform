"""
CodeIntel Platform — LangGraph ReAct Agent
Stateful hierarchical code intelligence agent with multi-source context fusion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic_settings import BaseSettings

from services.agent.reranker import RankedChunk, get_reranker
from services.agent.tools import ALL_TOOLS
from services.llm import LLMProvider, get_provider

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings (agent-specific, LLM config lives in services.llm.config)
# ---------------------------------------------------------------------------


class AgentSettings(BaseSettings):
    """Agent-specific configuration loaded from environment."""

    agent_max_steps: int = 10
    agent_rerank_top_k: int = 5
    agent_checkpoint_db: str = "/data/indexes/agent_checkpoints.db"

    class Config:
        env_file = ".env"
        case_sensitive = False


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------


class CodeSearchState(TypedDict):
    """
    LangGraph state for the CodeIntel ReAct agent.

    All fields are accumulated across steps. The agent reads and writes
    to this state at each reasoning step.
    """

    # Core conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Query context
    query: str
    project_id: str

    # Retrieved knowledge
    retrieved_chunks: list[dict[str, Any]]       # All raw chunks retrieved via tools
    reranked_chunks: list[dict[str, Any]]         # Cross-encoder ranked top-k chunks

    # Localization trace
    graph_path: list[str]                         # Traversal path (file → function → line)

    # Source attribution
    source_links: list[dict[str, Any]]           # Formatted source citations

    # Collection hit tracking
    collection_hits: dict[str, int]              # {"code_repo": N, "app_docs": M, "incident_reports": K}

    # Final output
    final_answer: str
    confidence: float

    # Control
    step_count: int
    tool_calls_made: list[str]


# ---------------------------------------------------------------------------
# Response Data Class
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """
    Structured response from the CodeIntel agent.

    Contains the answer, all source citations, and metadata about
    how the answer was produced.
    """

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    collection_hits: dict[str, int] = field(default_factory=dict)
    graph_path: list[str] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are CodeIntel, an expert AI assistant for on-premises code intelligence.
You have access to three knowledge collections:

1. **code_repo** — Live code from GitLab repositories (trust_level: code)
   - Use for: finding functions, classes, implementations, code patterns
   - Tools: search_code, keyword_search, traverse_graph, retrieve_entity

2. **app_docs** — Product and feature documentation from Application_documentation.md (trust_level: documentation)
   - Also contains AI-generated module documentation (trust_level: generated)
   - Use for: understanding what a feature does from a product perspective, API contracts, data models
   - Tools: search_app_docs

3. **incident_reports** — Historical incident post-mortems (trust_level: incident_report)
   - May be empty — this is NORMAL. If search_incidents returns [], continue without it.
   - Use for: past failure patterns, root cause analysis during on-call investigations
   - Tools: search_incidents

## Hierarchical Localization Protocol

For every query, you MUST follow this localization sequence:

**Step 1 — Repository Level**: Use search_code to find relevant code across the repository.
  Identify which files and modules are most relevant.

**Step 2 — File Level**: Narrow down to specific files using search_code with file_path filter,
  or keyword_search for exact names. Identify the specific file(s) containing the answer.

**Step 3 — Function Level**: Use retrieve_entity to get the exact function/class implementation.
  Or use traverse_graph to understand callers/callees if impact analysis is needed.

**Step 4 — Line Level**: Your final answer MUST cite exact line numbers (start_line, end_line)
  and include the GitLab permalink from the retrieved metadata.

## Multi-Source Integration Rules

- Always search code_repo first for code-level questions
- Always search app_docs for feature/behavior questions (product context enriches code answers)
- Search incident_reports when the query involves errors, failures, outages, or debugging
- Label every cited source: [code_repo], [app_docs], or [incident_reports]
- If sources contradict each other, trust code_repo (code is ground truth) over app_docs,
  which takes precedence over generated docs

## Trust Level Priority
code > documentation > generated > incident_report

## Answer Format

Your final answer must include:
1. A direct, precise answer to the question
2. The relevant code snippet (if code was found)
3. Source citations in this format:
   → `file_path` lines L{start}-L{end} [{collection}] — {gitlab_permalink}
4. Confidence level (high/medium/low) based on how directly the found code answers the question

## Important Rules
- Never hallucinate code that wasn't in the retrieved results
- If you cannot find relevant code, say so clearly — do not guess
- For incident queries, always note when incident_reports returns no results
- Always use the tools before answering — do not rely on general knowledge about the codebase
"""


# ---------------------------------------------------------------------------
# Agent Construction
# ---------------------------------------------------------------------------


class CodeIntelAgent:
    """
    LangGraph ReAct agent for code intelligence queries.

    Uses a stateful graph with tool nodes, cross-encoder reranking,
    and hierarchical localization to answer developer questions about code.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        """
        Initialize the CodeIntel agent with LLM, tools, and checkpointer.

        Args:
            llm: LLM provider to use.  Falls back to the global singleton
                 from ``services.llm.get_provider()`` if not supplied.
        """
        self._cfg = AgentSettings()
        self._llm_provider = llm or get_provider()
        self._reranker = get_reranker()
        self._log = structlog.get_logger(__name__)
        self._graph = None
        self._checkpointer = None

    def _get_llm(self, **kwargs):
        """
        Get a LangChain-compatible chat model from the LLM provider.

        Provider-agnostic: works with Ollama, llama.cpp, Bedrock, Azure.
        Returns whatever BaseChatModel the active provider supplies.
        """
        defaults = {"temperature": 0.1}  # Low temperature for precise code answers
        defaults.update(kwargs)
        return self._llm_provider.get_langchain_chat_model(**defaults)

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state graph."""
        llm = self._get_llm()
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        # Tool node
        tool_node = ToolNode(ALL_TOOLS)

        def agent_node(state: CodeSearchState) -> dict[str, Any]:
            """
            Main agent reasoning node.

            Invokes the LLM with the current message history and available tools.
            Appends the LLM response to the message list.
            """
            messages = state["messages"]

            # Enforce step budget
            step_count = state.get("step_count", 0) + 1
            if step_count > self._cfg.agent_max_steps:
                self._log.warning(
                    "agent_max_steps_exceeded",
                    step_count=step_count,
                    max_steps=self._cfg.agent_max_steps,
                )
                forced_answer = AIMessage(
                    content="I have reached the maximum number of reasoning steps. "
                    "Based on the information gathered so far, I will provide my best answer."
                )
                return {"messages": [forced_answer], "step_count": step_count}

            self._log.debug("agent_step", step=step_count)

            response = llm_with_tools.invoke(messages)
            return {"messages": [response], "step_count": step_count}

        def should_continue(state: CodeSearchState) -> str:
            """
            Router: decide whether to call tools or end the reasoning loop.

            Returns "tools" if the LLM wants to call tools, "rerank_and_answer"
            if the LLM produced a final answer, or "rerank_and_answer" if
            max steps exceeded.
            """
            messages = state["messages"]
            last_message = messages[-1] if messages else None

            if last_message is None:
                return "end"

            # Check if step budget exceeded
            if state.get("step_count", 0) >= self._cfg.agent_max_steps:
                return "rerank_and_answer"

            # If the last message has tool calls, continue to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Otherwise, final answer phase
            return "rerank_and_answer"

        def rerank_and_answer_node(state: CodeSearchState) -> dict[str, Any]:
            """
            Post-tool-call node: rerank retrieved chunks and generate final answer.

            Aggregates all chunks retrieved during tool calls, runs cross-encoder
            reranking, and generates the final cited answer.
            """
            # Extract all retrieved chunks from tool messages
            all_chunks: list[dict[str, Any]] = []
            tool_calls_made: list[str] = []
            collection_hits: dict[str, int] = {"code_repo": 0, "app_docs": 0, "incident_reports": 0}

            for msg in state["messages"]:
                if hasattr(msg, "name") and hasattr(msg, "content"):
                    # This is a ToolMessage
                    tool_calls_made.append(getattr(msg, "name", "unknown"))
                    try:
                        import json
                        content = msg.content
                        if isinstance(content, str):
                            tool_results = json.loads(content)
                            if isinstance(tool_results, list):
                                for r in tool_results:
                                    if isinstance(r, dict) and "chunk_id" in r:
                                        all_chunks.append(r)
                                        coll = r.get("collection", "code_repo")
                                        if coll in collection_hits:
                                            collection_hits[coll] += 1
                    except (json.JSONDecodeError, Exception):
                        pass

            # Cross-encoder reranking
            query = state.get("query", "")
            reranked: list[RankedChunk] = []
            if all_chunks and query:
                try:
                    reranked = self._reranker.rerank(
                        query=query,
                        chunks=all_chunks,
                        top_k=self._cfg.agent_rerank_top_k,
                    )
                except Exception as exc:
                    self._log.warning("reranking_failed_using_top_k", error=str(exc))
                    reranked = [
                        RankedChunk(chunk=c, score=c.get("score", 0.0), rank=i + 1)
                        for i, c in enumerate(all_chunks[: self._cfg.agent_rerank_top_k])
                    ]

            # Build source links for citation
            source_links = []
            for rc in reranked:
                source_links.append({
                    "file": rc.file_path or rc.chunk.get("heading", ""),
                    "symbol": rc.symbol_name,
                    "start_line": rc.start_line,
                    "end_line": rc.chunk.get("end_line", rc.start_line),
                    "collection": rc.collection,
                    "permalink": rc.gitlab_permalink,
                    "score": round(rc.score, 4),
                    "section_path": rc.chunk.get("section_path", ""),
                })

            # Build context string for final answer generation
            reranked_content = self._build_context_string(reranked)

            # Determine confidence from reranker scores
            confidence = 0.0
            if reranked:
                top_score = reranked[0].score
                # Cross-encoder scores are unbounded; normalize to 0-1 range roughly
                confidence = min(1.0, max(0.0, (top_score + 10) / 20))

            # Add reranked context as a system message for final answer
            context_msg = SystemMessage(
                content=(
                    f"## Reranked Evidence (top {len(reranked)} results)\n\n"
                    f"{reranked_content}\n\n"
                    "Now provide your final answer using ONLY the evidence above. "
                    "Cite sources with exact file paths, line numbers, and collection labels."
                )
            )

            # Final LLM call for answer synthesis
            final_messages = list(state["messages"]) + [context_msg]
            try:
                llm = self._get_llm()
                final_response = llm.invoke(final_messages)
                final_answer = final_response.content
            except Exception as exc:
                self._log.error("final_answer_generation_failed", error=str(exc))
                final_answer = (
                    "I encountered an error generating the final answer. "
                    "Please review the retrieved context above."
                )

            return {
                "messages": [AIMessage(content=final_answer)],
                "retrieved_chunks": all_chunks,
                "reranked_chunks": [rc.chunk for rc in reranked],
                "source_links": source_links,
                "collection_hits": collection_hits,
                "final_answer": final_answer,
                "confidence": confidence,
                "tool_calls_made": list(set(tool_calls_made)),
            }

        # Build the graph
        graph = StateGraph(CodeSearchState)

        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_node("rerank_and_answer", rerank_and_answer_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "rerank_and_answer": "rerank_and_answer",
                "end": END,
            },
        )

        graph.add_edge("tools", "agent")
        graph.add_edge("rerank_and_answer", END)

        # Setup SQLite checkpointer for durable state
        os.makedirs(os.path.dirname(self._cfg.agent_checkpoint_db), exist_ok=True)
        self._checkpointer = SqliteSaver.from_conn_string(
            self._cfg.agent_checkpoint_db
        )

        return graph.compile(checkpointer=self._checkpointer)

    def _build_context_string(self, reranked: list[RankedChunk]) -> str:
        """Build a formatted context string from reranked chunks for the LLM."""
        if not reranked:
            return "No relevant context found."

        parts = []
        for rc in reranked:
            chunk = rc.chunk
            coll = rc.collection
            file_path = rc.file_path or chunk.get("heading", "")
            symbol = rc.symbol_name
            start = rc.start_line
            end = chunk.get("end_line", start)
            permalink = rc.gitlab_permalink
            content = rc.content

            section_path = chunk.get("section_path", "")
            heading = chunk.get("heading", "")

            header_parts = [f"[SOURCE: {coll}]"]
            if file_path:
                header_parts.append(f"file: {file_path}")
            if symbol:
                header_parts.append(f"symbol: {symbol}")
            if start:
                header_parts.append(f"lines: {start}-{end}")
            if section_path:
                header_parts.append(f"section: {section_path}")
            if permalink:
                header_parts.append(f"permalink: {permalink}")

            parts.append(" | ".join(header_parts) + "\n" + content)

        return "\n\n---\n\n".join(parts)

    def query(
        self,
        question: str,
        project_id: str = "",
        thread_id: str | None = None,
    ) -> AgentResponse:
        """
        Execute a code intelligence query through the ReAct agent.

        Args:
            question: Natural language question about the codebase.
            project_id: GitLab project ID to scope the search.
            thread_id: Optional thread ID for conversation continuity (checkpointing).

        Returns:
            AgentResponse with answer, sources, and metadata.
        """
        if self._graph is None:
            self._graph = self._build_graph()

        # Build initial state
        initial_state: CodeSearchState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Project: {project_id}\n\nQuestion: {question}"),
            ],
            "query": question,
            "project_id": project_id,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "graph_path": [],
            "source_links": [],
            "collection_hits": {"code_repo": 0, "app_docs": 0, "incident_reports": 0},
            "final_answer": "",
            "confidence": 0.0,
            "step_count": 0,
            "tool_calls_made": [],
        }

        config = {"configurable": {"thread_id": thread_id or f"{project_id}:{question[:32]}"}}

        self._log.info(
            "agent_query_started",
            question=question[:80],
            project_id=project_id,
        )

        try:
            final_state = self._graph.invoke(initial_state, config=config)
        except Exception as exc:
            self._log.error("agent_query_failed", error=str(exc), question=question[:80])
            raise

        self._log.info(
            "agent_query_completed",
            question=question[:80],
            confidence=final_state.get("confidence", 0.0),
            sources_found=len(final_state.get("source_links", [])),
        )

        return AgentResponse(
            answer=final_state.get("final_answer", ""),
            sources=final_state.get("source_links", []),
            confidence=final_state.get("confidence", 0.0),
            collection_hits=final_state.get("collection_hits", {}),
            graph_path=final_state.get("graph_path", []),
            tool_calls_made=final_state.get("tool_calls_made", []),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_agent_instance: CodeIntelAgent | None = None


def get_agent() -> CodeIntelAgent:
    """Return the module-level agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CodeIntelAgent()
    return _agent_instance
