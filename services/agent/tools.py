"""
CodeIntel Platform — LangGraph Agent Tools
All 6 tools available to the ReAct agent for code intelligence queries.
"""

from __future__ import annotations

import os
from typing import Any

import structlog
from langchain_core.tools import tool
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from services.indexing.embedder import get_embedder
from services.indexing.qdrant_store import get_store

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class ToolSettings(BaseSettings):
    """Agent tool configuration loaded from environment."""

    agent_search_top_k: int = 20
    qdrant_collection_code: str = "code_repo"
    qdrant_collection_app_docs: str = "app_docs"
    qdrant_collection_incidents: str = "incident_reports"
    gitlab_url: str = "https://gitlab.com"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


_settings = ToolSettings()
_embedder = None
_store = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _get_store():
    global _store
    if _store is None:
        _store = get_store()
    return _store


def reset_tool_singletons() -> None:
    """Reset cached singletons — intended for test setup/teardown only."""
    global _embedder, _store
    _embedder = None
    _store = None


# ---------------------------------------------------------------------------
# Result Serialization
# ---------------------------------------------------------------------------


def _get_permalink(meta: dict[str, Any]) -> str:
    """Extract the SCM permalink from chunk metadata, with legacy fallback."""
    # DEPRECATED fallback: "gitlab_permalink" kept until all indexed data
    # has been re-embedded with the newer "scm_permalink" key.
    return meta.get("scm_permalink", "") or meta.get("gitlab_permalink", "")


def _format_result(hit: dict[str, Any]) -> dict[str, Any]:
    """
    Format a raw Qdrant search result into a clean agent-facing dict.

    Returns a structured dict the LLM can reason over, including source
    attribution and SCM permalink.
    """
    meta = hit.get("metadata", {})
    permalink = _get_permalink(meta)
    # Truncate content to 500 chars to prevent context overflow in the LLM
    content = hit.get("content", meta.get("content", ""))
    if len(content) > 500:
        content = content[:500] + "…"
    return {
        "chunk_id": hit.get("chunk_id", ""),
        "score": round(hit.get("score", 0.0), 4),
        "collection": hit.get("collection", meta.get("source_collection", "unknown")),
        "trust_level": meta.get("trust_level", "unknown"),
        # Code-specific fields
        "file_path": meta.get("file_path", ""),
        "language": meta.get("language", ""),
        "symbol_name": meta.get("symbol_name", ""),
        "symbol_type": meta.get("symbol_type", ""),
        "qualified_name": meta.get("qualified_name", ""),
        "parent_symbol": meta.get("parent_symbol", ""),
        "start_line": meta.get("start_line", 0),
        "end_line": meta.get("end_line", 0),
        "scm_permalink": permalink,
        "gitlab_permalink": permalink,  # backwards compat
        "project_id": meta.get("project_id", ""),
        "branch": meta.get("branch", ""),
        "commit_sha": meta.get("commit_sha", ""),
        # Doc-specific fields
        "section_path": meta.get("section_path", ""),
        "heading": meta.get("heading", ""),
        "source_label": meta.get("source_label", ""),
        # Content — truncated to 500 chars to prevent LLM context overflow
        "content": content,
        # Incident-specific fields
        "severity": meta.get("severity", ""),
        "incident_date": meta.get("incident_date", ""),
        "affected_services": meta.get("affected_services", []),
    }


# ---------------------------------------------------------------------------
# Tool 1: search_code
# ---------------------------------------------------------------------------


@tool
def search_code(
    query: str,
    language: str | None = None,
    file_path: str | None = None,
    project_id: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Search the code repository for semantically relevant code snippets.

    Use this tool when you need to find:
    - Functions or classes that implement specific behavior
    - Code related to a particular feature or bug
    - How something is implemented at the code level
    - All code in a specific file or by language

    The search uses dense vector similarity (CodeRankEmbed) combined with
    BM25 lexical matching for best retrieval.

    Args:
        query: Natural language description of what you're looking for.
            Examples: "payment retry logic", "database connection pooling",
            "authentication middleware", "error handling in order processing"
        language: Optional language filter (python, javascript, typescript,
            go, java, c, cpp). Set to narrow results to a specific language.
        file_path: Optional file path filter (partial match).
            Set to narrow results to a specific file.
        project_id: Optional project ID filter.
        top_k: Number of results to return (default 20, max 50).

    Returns:
        List of code chunk dicts, each containing:
        - file_path, language, symbol_name, symbol_type
        - start_line, end_line (1-indexed)
        - scm_permalink / gitlab_permalink (direct link to the code)
        - content (the actual code)
        - score (relevance score 0-1)
        - collection: always "code_repo"
    """
    embedder = _get_embedder()
    store = _get_store()

    query_vector = embedder.embed_query(query)

    filters: dict[str, Any] = {}
    if language:
        filters["language"] = language
    if file_path:
        filters["file_path"] = file_path
    if project_id:
        filters["project_id"] = project_id

    results = store.hybrid_search(
        collection=_settings.qdrant_collection_code,
        query_vector=query_vector,
        query_text=query,
        filters=filters if filters else None,
        top_k=min(top_k, 50),
    )

    log.debug(
        "search_code_results",
        query=query[:80],
        results_count=len(results),
        language=language,
        file_path=file_path,
    )

    return [_format_result(r) for r in results]


# ---------------------------------------------------------------------------
# Tool 2: search_app_docs
# ---------------------------------------------------------------------------


@tool
def search_app_docs(
    query: str,
    section_path: str | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Search the Application_documentation.md for product/feature context.

    Use this tool when you need to understand:
    - What a feature does from a product/business perspective
    - API contracts, data models, or user-facing behavior
    - Product requirements or specifications
    - Auto-generated module documentation

    This collection covers both the manually written Application_documentation.md
    and AI-generated documentation produced by the doc-gen pipeline.

    Args:
        query: Natural language description of what you're looking for.
            Examples: "payment processing flow", "user authentication API",
            "order state machine", "rate limiting policy"
        section_path: Optional section filter (e.g., "API / Authentication").
        top_k: Number of results to return (default 10).

    Returns:
        List of documentation chunk dicts, each containing:
        - heading, section_path (document structure)
        - content (the documentation text)
        - source_label ("application_docs" or "generated")
        - trust_level ("documentation" or "generated")
        - score (relevance score)
        - collection: always "app_docs"
    """
    embedder = _get_embedder()
    store = _get_store()

    query_vector = embedder.embed_query(query)

    filters: dict[str, Any] = {}
    if section_path:
        filters["section_path"] = section_path

    results = store.hybrid_search(
        collection=_settings.qdrant_collection_app_docs,
        query_vector=query_vector,
        query_text=query,
        filters=filters if filters else None,
        top_k=min(top_k, 30),
    )

    log.debug(
        "search_app_docs_results",
        query=query[:80],
        results_count=len(results),
    )

    return [_format_result(r) for r in results]


# ---------------------------------------------------------------------------
# Tool 3: search_incidents
# ---------------------------------------------------------------------------


@tool
def search_incidents(
    query: str,
    severity: str | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Search historical incident reports for past failure patterns.

    Use this tool when you need to understand:
    - Whether a similar error or outage has happened before
    - Root causes of past incidents in this codebase
    - What remediation was applied to previous failures
    - Historical context for on-call investigation

    IMPORTANT: This collection may be empty if no incident_reports.md has been
    provided. If empty, this tool returns [] — this is normal and expected.
    The agent should continue without incident context in that case.

    Args:
        query: Natural language description of the incident pattern.
            Examples: "payment queue timeout", "database connection exhaustion",
            "memory leak in order processor", "authentication service down"
        severity: Optional severity filter (P1, P2, P3, P4).
        top_k: Number of results to return (default 10).

    Returns:
        List of incident chunk dicts (may be empty []), each containing:
        - heading (incident title), incident_date, severity
        - affected_services (list of service names)
        - content (incident description, root cause, resolution)
        - collection: always "incident_reports"
        Returns [] if the collection is empty or not configured.
    """
    embedder = _get_embedder()
    store = _get_store()

    # Graceful degradation: return [] if collection is empty
    if not store.is_collection_populated(_settings.qdrant_collection_incidents):
        log.debug(
            "incident_reports_collection_empty_returning_empty",
            collection=_settings.qdrant_collection_incidents,
        )
        return []

    query_vector = embedder.embed_query(query)

    filters: dict[str, Any] = {}
    if severity:
        filters["severity"] = severity.upper()

    results = store.hybrid_search(
        collection=_settings.qdrant_collection_incidents,
        query_vector=query_vector,
        query_text=query,
        filters=filters if filters else None,
        top_k=min(top_k, 30),
    )

    log.debug(
        "search_incidents_results",
        query=query[:80],
        results_count=len(results),
    )

    return [_format_result(r) for r in results]


# ---------------------------------------------------------------------------
# Tool 4: keyword_search
# ---------------------------------------------------------------------------


@tool
def keyword_search(
    terms: list[str],
    collection: str = "code_repo",
    project_id: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Hybrid (dense + BM25 text) search biased toward exact identifier matching.

    Use this tool when you need to find:
    - Exact function names, class names, or variable names
    - Specific error messages or log strings
    - Exact import statements or module references
    - Cases where you know the exact name and need to find all references

    Internally this uses the same hybrid search as search_code but the query
    is constructed from exact terms, making BM25 the dominant signal.
    This complements semantic search (search_code) for cases where exact
    token matching is more important than semantic similarity.

    Args:
        terms: List of exact terms to search for.
            Examples: ["retry_payment", "PaymentError"],
            ["ConnectionPool", "max_connections"],
            ["OutOfMemoryError", "heap"]
        collection: Which collection to search:
            "code_repo" (default), "app_docs", or "incident_reports"
        project_id: Optional project ID filter.
        top_k: Number of results to return (default 20).

    Returns:
        List of chunk dicts matching the search terms, same structure as search_code.
        Results are ranked by BM25 relevance (term frequency / document frequency).
    """
    embedder = _get_embedder()
    store = _get_store()

    # Build a combined query string from terms for BM25
    query_text = " ".join(terms)

    # Use the query embedding for hybrid search
    query_vector = embedder.embed_query(query_text)

    filters: dict[str, Any] = {}
    if project_id:
        filters["project_id"] = project_id

    # Map collection alias to actual name
    collection_map = {
        "code_repo": _settings.qdrant_collection_code,
        "app_docs": _settings.qdrant_collection_app_docs,
        "incident_reports": _settings.qdrant_collection_incidents,
    }
    actual_collection = collection_map.get(collection, collection)

    # Check if collection is populated (for optional collections)
    if not store.is_collection_populated(actual_collection):
        log.debug(
            "keyword_search_collection_empty",
            collection=actual_collection,
        )
        return []

    results = store.hybrid_search(
        collection=actual_collection,
        query_vector=query_vector,
        query_text=query_text,
        filters=filters if filters else None,
        top_k=min(top_k, 50),
    )

    log.debug(
        "keyword_search_results",
        terms=terms,
        collection=actual_collection,
        results_count=len(results),
    )

    return [_format_result(r) for r in results]


# ---------------------------------------------------------------------------
# Tool 5: traverse_graph
# ---------------------------------------------------------------------------


@tool
def traverse_graph(
    node_id: str,
    direction: str = "both",
    depth: int = 2,
    project_id: str | None = None,
) -> dict[str, Any]:
    """
    Traverse the code dependency graph to find callers and callees of a function.

    Use this tool when you need to understand:
    - What functions call a specific function (callers/upstream)
    - What functions a specific function calls (callees/downstream)
    - The full call chain for impact analysis before a refactor
    - Which code depends on a function you're about to change

    The node_id is the function's qualified name:
    - For a top-level function: "function_name"
    - For a class method: "ClassName.method_name"
    - With file context: "file_path:ClassName.method_name"

    Args:
        node_id: Qualified function name to traverse from.
            Examples: "PaymentService.process_payment",
            "retry_payment", "src/auth/middleware.py:verify_token"
        direction: Traversal direction:
            "callers" — find functions that call this function
            "callees" — find functions called by this function
            "both" — find both callers and callees (default)
        depth: Accepted for API compatibility but currently always traverses 1 hop.
            Multi-hop traversal is not yet implemented (GAP-06).
        project_id: Optional project ID to scope traversal.

    Returns:
        Dict with:
        - "node_id": the queried function
        - "callers": list of functions that call this function (with file info)
        - "callees": list of functions called by this function (with file info)
        - "depth": depth accepted (always 1 in current implementation)
    """
    store = _get_store()
    embedder = _get_embedder()
    depth = 1  # Multi-hop not yet implemented; clamp to avoid misleading the caller

    # Parse node_id to extract file_path and symbol name
    file_path_hint = None
    symbol_name = node_id

    if ":" in node_id:
        parts = node_id.split(":", 1)
        file_path_hint = parts[0]
        symbol_name = parts[1]

    # Clean up symbol name
    if "." in symbol_name:
        parts = symbol_name.split(".")
        base_name = parts[-1]
    else:
        base_name = symbol_name

    result: dict[str, Any] = {
        "node_id": node_id,
        "callers": [],
        "callees": [],
        "depth": depth,
    }

    filters: dict[str, Any] = {}
    if project_id:
        filters["project_id"] = project_id
    if file_path_hint:
        filters["file_path"] = file_path_hint

    # Find the target node first
    target_results = store.hybrid_search(
        collection=_settings.qdrant_collection_code,
        query_vector=embedder.embed_query(f"function {base_name}"),
        query_text=base_name,
        filters=filters if filters else None,
        top_k=5,
    )

    # Extract callees from the target node's metadata
    if direction in ("callees", "both") and target_results:
        target = target_results[0]
        callees = target.get("metadata", {}).get("calls", [])

        if callees:
            # Find chunks for each callee
            for callee_name in callees[:20]:  # limit traversal width
                callee_results = store.hybrid_search(
                    collection=_settings.qdrant_collection_code,
                    query_vector=embedder.embed_query(f"function {callee_name}"),
                    query_text=callee_name,
                    filters={"project_id": project_id} if project_id else None,
                    top_k=3,
                )
                if callee_results:
                    top = callee_results[0]
                    meta = top.get("metadata", {})
                    callee_link = _get_permalink(meta)
                    result["callees"].append({
                        "name": callee_name,
                        "file_path": meta.get("file_path", ""),
                        "start_line": meta.get("start_line", 0),
                        "qualified_name": meta.get("qualified_name", callee_name),
                        "scm_permalink": callee_link,
                        "gitlab_permalink": callee_link,  # DEPRECATED: use scm_permalink
                    })

    # Find callers: search for chunks that have this function in their calls list
    if direction in ("callers", "both"):
        # Search for chunks that reference the base name
        caller_results = store.hybrid_search(
            collection=_settings.qdrant_collection_code,
            query_vector=embedder.embed_query(f"calls {base_name}"),
            query_text=base_name,
            filters={"project_id": project_id} if project_id else None,
            top_k=20,
        )

        # Filter to actual callers (have the function name in their calls list)
        for r in caller_results:
            calls_list = r.get("metadata", {}).get("calls", [])
            if base_name in calls_list:
                meta = r.get("metadata", {})
                caller_link = _get_permalink(meta)
                result["callers"].append({
                    "name": meta.get("symbol_name", ""),
                    "qualified_name": meta.get("qualified_name", ""),
                    "file_path": meta.get("file_path", ""),
                    "start_line": meta.get("start_line", 0),
                    "scm_permalink": caller_link,
                    "gitlab_permalink": caller_link,  # DEPRECATED: use scm_permalink
                })

    log.debug(
        "graph_traversal_complete",
        node_id=node_id,
        callers_found=len(result["callers"]),
        callees_found=len(result["callees"]),
    )

    return result


# ---------------------------------------------------------------------------
# Tool 6: retrieve_entity
# ---------------------------------------------------------------------------


@tool
def retrieve_entity(
    node_id: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve the exact source code and metadata for a specific function or class.

    Use this tool when you have already identified the function you want and
    need to retrieve its complete source code with full metadata.

    This is the final step in hierarchical localization — you've narrowed to
    the function level and now need the exact content and permalink.

    The node_id can be specified as:
    - Symbol name: "retry_payment"
    - Qualified name: "PaymentService.retry_payment"
    - File-qualified: "src/payment/retry.py:retry_payment"
    - File-qualified with class: "src/payment/retry.py:PaymentService.retry_payment"

    Args:
        node_id: Symbol identifier. Preferably as specific as possible.
            The more specific, the more accurate the retrieval.
        project_id: Optional project ID to scope retrieval.

    Returns:
        Dict containing:
        - "found": bool (True if entity was found)
        - "file_path": relative path to the file
        - "symbol_name": function/class name
        - "qualified_name": ClassName.method_name if a method
        - "symbol_type": "function", "class", or "method"
        - "start_line": first line number (1-indexed)
        - "end_line": last line number
        - "content": the full source code of the function/class
        - "docstring": extracted docstring if present
        - "params": list of parameter names
        - "calls": list of function names called by this symbol
        - "scm_permalink": direct URL to this symbol (GitLab or GitHub)
        - "gitlab_permalink": alias of scm_permalink for backwards compat
        - "collection": "code_repo"
        - "project_id": project identifier
        - "commit_sha": commit SHA when this was indexed
    """
    store = _get_store()
    embedder = _get_embedder()

    # Parse the node_id
    file_path_hint = None
    symbol_name = node_id

    if ":" in node_id:
        parts = node_id.split(":", 1)
        file_path_hint = parts[0]
        symbol_name = parts[1]

    # Build search query
    base_name = symbol_name.split(".")[-1] if "." in symbol_name else symbol_name
    query = f"function {base_name} implementation"

    filters: dict[str, Any] = {}
    if project_id:
        filters["project_id"] = project_id
    if file_path_hint:
        filters["file_path"] = file_path_hint

    # Try exact symbol_name match first
    results = store.hybrid_search(
        collection=_settings.qdrant_collection_code,
        query_vector=embedder.embed_query(query),
        query_text=base_name,
        filters=filters if filters else None,
        top_k=10,
    )

    # Find the best match by symbol name
    best_match = None
    for r in results:
        meta = r.get("metadata", {})
        sym = meta.get("symbol_name", "")
        qual = meta.get("qualified_name", "")

        # Exact match preferred
        if sym == base_name or qual == symbol_name:
            best_match = r
            break

        # Partial match as fallback
        if best_match is None and (base_name.lower() in sym.lower() or base_name.lower() in qual.lower()):
            best_match = r

    if best_match is None:
        log.info("entity_not_found", node_id=node_id)
        return {
            "found": False,
            "node_id": node_id,
            "message": f"No entity found matching '{node_id}'",
        }

    meta = best_match.get("metadata", {})
    content = best_match.get("content", meta.get("content", ""))
    # Truncate to 2000 chars — intentionally larger than search_code's 500-char
    # limit since retrieve_entity is an explicit full-source retrieval, but still
    # bounded to prevent LLM context overflow for very large functions (GAP-07).
    if len(content) > 2000:
        content = content[:2000] + "…"

    result = {
        "found": True,
        "chunk_id": best_match.get("chunk_id", ""),
        "file_path": meta.get("file_path", ""),
        "symbol_name": meta.get("symbol_name", ""),
        "qualified_name": meta.get("qualified_name", ""),
        "symbol_type": meta.get("symbol_type", ""),
        "parent_symbol": meta.get("parent_symbol", ""),
        "start_line": meta.get("start_line", 0),
        "end_line": meta.get("end_line", 0),
        "language": meta.get("language", ""),
        "content": content,
        "docstring": meta.get("docstring", ""),
        "params": meta.get("params", []),
        "calls": meta.get("calls", []),
        "scm_permalink": _get_permalink(meta),
        "gitlab_permalink": _get_permalink(meta),  # DEPRECATED: use scm_permalink
        "project_id": meta.get("project_id", ""),
        "branch": meta.get("branch", ""),
        "commit_sha": meta.get("commit_sha", ""),
        "collection": "code_repo",
        "trust_level": "code",
    }

    log.debug(
        "entity_retrieved",
        node_id=node_id,
        file_path=result["file_path"],
        start_line=result["start_line"],
    )

    return result


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_code,
    search_app_docs,
    search_incidents,
    keyword_search,
    traverse_graph,
    retrieve_entity,
]
