"""
CodeIntel Platform — LangGraph ReAct Agent
Stateful hierarchical code intelligence agent with multi-source context fusion.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ConfigDict
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

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


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

SYSTEM_PROMPT = """You are CodeIntel, an AI assistant for code intelligence queries.

You have access to these tools:
- search_code: semantic search over indexed code (use first, always)
- keyword_search: exact name/symbol lookup  
- retrieve_entity: fetch a specific function/class by name
- traverse_graph: find callers/callees of a function
- search_app_docs: search product documentation
- search_incidents: search incident post-mortems (may be empty — that is normal)

Instructions:
1. Always call search_code first with a descriptive query.
2. Call keyword_search if you need exact symbol names.
3. After retrieving results, provide a concise answer with file paths and line numbers.
4. Cite sources as: `file_path` lines L{start}-L{end}
5. If no results found, say so clearly — do not guess.
6. Keep answers brief and focused.
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
            import json as _json

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

            # Some quantized models (e.g. qwen2.5-coder Q4_K_M) emit tool calls as
            # a JSON string in the content field instead of the structured tool_calls
            # list that LangChain/Ollama normally produce.  Detect and fix that here.
            if not getattr(response, "tool_calls", None) and getattr(response, "content", None):
                content = response.content.strip()
                # Strip markdown code fences if present
                if content.startswith("```"):
                    content = "\n".join(
                        line for line in content.splitlines()
                        if not line.startswith("```")
                    ).strip()
                try:
                    parsed = _json.loads(content)
                    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        self._log.debug(
                            "agent_tool_call_from_content",
                            tool=parsed["name"],
                        )
                        response = AIMessage(
                            content="",
                            tool_calls=[{
                                "id": f"call_{parsed['name']}_{step_count}",
                                "name": parsed["name"],
                                "args": parsed.get("arguments") or {},
                                "type": "tool_call",
                            }],
                        )
                except (_json.JSONDecodeError, ValueError, TypeError, KeyError):
                    # Only catch expected JSON/type errors — let real bugs propagate
                    pass

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

            # Check if step budget exceeded (mirrors the > check in agent_node)
            if state.get("step_count", 0) > self._cfg.agent_max_steps:
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
                    # This is a ToolMessage — name is None on non-tool messages
                    msg_name = getattr(msg, "name", None)
                    if msg_name is None:
                        continue
                    tool_calls_made.append(msg_name)
                    try:
                        import json
                        content = msg.content
                        if isinstance(content, str):
                            tool_results = json.loads(content)
                            if isinstance(tool_results, list):
                                # search_code / search_app_docs / search_incidents / keyword_search
                                for r in tool_results:
                                    if isinstance(r, dict) and "chunk_id" in r:
                                        all_chunks.append(r)
                                        coll = r.get("collection", "code_repo")
                                        if coll in collection_hits:
                                            collection_hits[coll] += 1
                            elif isinstance(tool_results, dict):
                                # retrieve_entity — has content + metadata directly
                                if tool_results.get("found") and tool_results.get("content"):
                                    chunk = {
                                        "chunk_id": tool_results.get("chunk_id", f"entity_{msg_name}"),
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
                                # traverse_graph — summarise callers/callees as a text chunk
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

        # Setup SQLite checkpointer for durable state.
        # Use WAL mode so concurrent readers from multiple Gunicorn workers do not
        # block each other.  Each process opens its own connection (not shared
        # across fork boundaries) to avoid write-corruption in a pre-fork model.
        os.makedirs(os.path.dirname(self._cfg.agent_checkpoint_db), exist_ok=True)
        conn = sqlite3.connect(
            self._cfg.agent_checkpoint_db,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._checkpointer = SqliteSaver(conn=conn)

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
