"""
CodeIntel Platform — Agent FastAPI Service
REST API for the LangGraph ReAct code intelligence agent with SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from services.agent.agent import AgentResponse, get_agent
from services.indexing.qdrant_store import get_store

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Wiki helpers (lazy-loaded so agent service doesn't crash without wiki dir)
# ---------------------------------------------------------------------------

_wiki_base: str | None = None


def _get_wiki_base() -> str:
    global _wiki_base
    if _wiki_base is None:
        _wiki_base = os.environ.get("WIKI_DOCS_DIR", "/data/indexes/wiki")
    return _wiki_base

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class APISettings(BaseSettings):
    """Agent API settings."""

    log_level: str = "info"
    secret_key: str = "change-me-in-production"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = APISettings()

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), settings.log_level.upper(), 20)
    ),
    logger_factory=structlog.PrintLoggerFactory(),
)

# ---------------------------------------------------------------------------
# Lifespan — initialise singletons at startup (avoids race conditions and
# cold-start latency on first user request)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up all heavy singletons before the server accepts traffic."""
    from services.agent.reranker import get_reranker
    from services.indexing.embedder import get_embedder

    log.info("lifespan_startup", msg="Initialising singletons…")

    # Qdrant store (lightweight — just a TCP connection)
    app.state.store = get_store()

    # Embedder — loads the sentence-transformers model from disk/cache
    app.state.embedder = get_embedder()

    # Cross-encoder reranker — loads ms-marco-MiniLM-L6-v2
    app.state.reranker = get_reranker()

    # LangGraph agent — builds the state graph and initialises the LLM
    agent = get_agent()
    agent._build_graph()
    app.state.agent = agent

    log.info("lifespan_startup_complete", msg="All singletons ready")
    yield
    log.info("lifespan_shutdown")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CodeIntel Agent API",
    description="AI code intelligence agent with multi-source RAG and hierarchical localization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the /agent/query endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    project_id: str = Field("", description="GitLab project ID to scope the search")
    thread_id: str | None = Field(None, description="Thread ID for conversation continuity")
    stream: bool = Field(False, description="If True, stream response via SSE")


class LocalizeRequest(BaseModel):
    """Request body for the /agent/localize endpoint."""

    issue_text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Issue description, error message, or bug report to localize",
    )
    project_id: str = Field("", description="GitLab project ID to scope the search")
    severity: str | None = Field(None, description="Optional severity hint (P1/P2/P3/P4)")


class SourceCitation(BaseModel):
    """A single source citation in the agent response."""

    file: str = ""
    symbol: str = ""
    start_line: int = 0
    end_line: int = 0
    collection: str = ""
    permalink: str = ""
    score: float = 0.0
    section_path: str = ""


class QueryResponse(BaseModel):
    """Response from the /agent/query endpoint."""

    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: float = 0.0
    collection_hits: dict[str, int] = Field(default_factory=dict)
    tool_calls_made: list[str] = Field(default_factory=list)


class LocalizeResponse(BaseModel):
    """Response from the /agent/localize endpoint."""

    localized: bool
    file_path: str = ""
    function_name: str = ""
    start_line: int = 0
    end_line: int = 0
    permalink: str = ""
    confidence: float = 0.0
    explanation: str = ""
    sources: list[SourceCitation] = Field(default_factory=list)


class CollectionStatus(BaseModel):
    """Status of a single Qdrant collection."""

    collection: str
    points_count: int
    status: str
    exists: bool
    is_populated: bool


class CollectionsStatusResponse(BaseModel):
    """Response from the /agent/collections/status endpoint."""

    code_repo: CollectionStatus
    app_docs: CollectionStatus
    incident_reports: CollectionStatus


# ---------------------------------------------------------------------------
# Helper: Format AgentResponse → QueryResponse
# ---------------------------------------------------------------------------


def _format_response(agent_resp: AgentResponse) -> QueryResponse:
    """Convert AgentResponse to the QueryResponse API model."""
    sources = [
        SourceCitation(
            file=s.get("file", ""),
            symbol=s.get("symbol", ""),
            start_line=s.get("start_line", 0),
            end_line=s.get("end_line", 0),
            collection=s.get("collection", ""),
            permalink=s.get("permalink", ""),
            score=s.get("score", 0.0),
            section_path=s.get("section_path", ""),
        )
        for s in agent_resp.sources
    ]
    return QueryResponse(
        answer=agent_resp.answer,
        sources=sources,
        confidence=agent_resp.confidence,
        collection_hits=agent_resp.collection_hits,
        tool_calls_made=agent_resp.tool_calls_made,
    )


# ---------------------------------------------------------------------------
# SSE Streaming Helper
# ---------------------------------------------------------------------------


async def _stream_agent_response(
    query: str,
    project_id: str,
    thread_id: str | None,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for a streaming agent query.

    Yields events in the format:
    - data: {"type": "thinking", "content": "..."}\n\n
    - data: {"type": "answer", "content": "..."}\n\n
    - data: {"type": "sources", "content": [...]}\n\n
    - data: {"type": "done"}\n\n
    """
    agent = get_agent()

    yield f"data: {json.dumps({'type': 'thinking', 'content': 'Searching knowledge base...'})}\n\n"
    await asyncio.sleep(0)

    try:
        # Run in thread pool since the agent is synchronous
        loop = asyncio.get_event_loop()
        agent_resp = await loop.run_in_executor(
            None,
            lambda: agent.query(question=query, project_id=project_id, thread_id=thread_id),
        )

        # Stream the answer word by word for a better UX
        words = agent_resp.answer.split(" ")
        chunk_size = 5
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
            await asyncio.sleep(0.01)

        # Send sources
        sources = [
            {
                "file": s.get("file", ""),
                "symbol": s.get("symbol", ""),
                "start_line": s.get("start_line", 0),
                "end_line": s.get("end_line", 0),
                "collection": s.get("collection", ""),
                "permalink": s.get("permalink", ""),
            }
            for s in agent_resp.sources
        ]
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

        # Send metadata
        metadata = {
            "confidence": agent_resp.confidence,
            "collection_hits": agent_resp.collection_hits,
        }
        yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"

    except Exception as exc:
        log.error("streaming_agent_error", error=str(exc))
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Infrastructure"])
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns service status, version, and Qdrant connectivity status.
    """
    store = get_store()
    qdrant_ok = False
    try:
        stats = store.get_all_collection_stats()
        qdrant_ok = True
    except Exception:
        stats = {}

    return {
        "status": "healthy",
        "service": "agent",
        "version": "1.0.0",
        "qdrant_connected": qdrant_ok,
    }


@app.post(
    "/agent/query",
    response_model=QueryResponse,
    tags=["Agent"],
)
async def agent_query(request: QueryRequest) -> Any:
    """
    Execute a natural language code intelligence query.

    The agent searches across all 3 knowledge collections (code_repo, app_docs,
    incident_reports), reranks results with cross-encoder, and generates
    a cited answer with exact file/function/line attribution.

    Set `stream: true` to receive a Server-Sent Events streaming response.
    """
    log.info(
        "agent_query_received",
        query=request.query[:80],
        project_id=request.project_id,
        stream=request.stream,
    )

    if request.stream:
        return StreamingResponse(
            _stream_agent_response(
                query=request.query,
                project_id=request.project_id,
                thread_id=request.thread_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: run agent and return full response
    agent = get_agent()
    try:
        loop = asyncio.get_event_loop()
        agent_resp = await loop.run_in_executor(
            None,
            lambda: agent.query(
                question=request.query,
                project_id=request.project_id,
                thread_id=request.thread_id,
            ),
        )
    except Exception as exc:
        log.error("agent_query_failed", error=str(exc), query=request.query[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent query failed: {str(exc)}",
        )

    return _format_response(agent_resp)


@app.post(
    "/agent/localize",
    response_model=LocalizeResponse,
    tags=["Agent"],
)
async def agent_localize(request: LocalizeRequest) -> LocalizeResponse:
    """
    Localize an issue description to exact code locations.

    Takes a bug report, error message, or issue description and returns
    the most likely code location (file, function, line) responsible.

    This is the primary endpoint for on-call engineers during incidents.
    """
    log.info(
        "agent_localize_received",
        issue_text=request.issue_text[:80],
        project_id=request.project_id,
    )

    # Build a localization-specific query
    localize_query = (
        f"Localize this issue to the exact code location:\n\n{request.issue_text}\n\n"
        f"Find the specific file, function, and line numbers responsible. "
        f"Also search incident_reports for similar past incidents."
    )

    if request.severity:
        localize_query += f"\n\nSeverity: {request.severity}"

    agent = get_agent()
    try:
        loop = asyncio.get_event_loop()
        agent_resp = await loop.run_in_executor(
            None,
            lambda: agent.query(
                question=localize_query,
                project_id=request.project_id,
            ),
        )
    except Exception as exc:
        log.error("agent_localize_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Localization failed: {str(exc)}",
        )

    # Extract the primary localization from sources
    primary_source = agent_resp.sources[0] if agent_resp.sources else {}
    code_sources = [
        s for s in agent_resp.sources
        if s.get("collection") == "code_repo"
    ]
    top_code = code_sources[0] if code_sources else primary_source

    return LocalizeResponse(
        localized=bool(top_code),
        file_path=top_code.get("file", ""),
        function_name=top_code.get("symbol", ""),
        start_line=top_code.get("start_line", 0),
        end_line=top_code.get("end_line", 0),
        permalink=top_code.get("permalink", ""),
        confidence=agent_resp.confidence,
        explanation=agent_resp.answer,
        sources=[
            SourceCitation(
                file=s.get("file", ""),
                symbol=s.get("symbol", ""),
                start_line=s.get("start_line", 0),
                end_line=s.get("end_line", 0),
                collection=s.get("collection", ""),
                permalink=s.get("permalink", ""),
                score=s.get("score", 0.0),
            )
            for s in agent_resp.sources
        ],
    )


@app.get(
    "/agent/collections/status",
    response_model=CollectionsStatusResponse,
    tags=["Agent"],
)
async def collections_status() -> CollectionsStatusResponse:
    """
    Get the status and point counts for all 3 Qdrant collections.

    Returns whether each collection exists, how many points are indexed,
    and the Qdrant collection status. Use to verify the system is ready.
    """
    store = get_store()

    try:
        all_stats = store.get_all_collection_stats()
    except Exception as exc:
        log.error("collections_status_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve collection stats: {str(exc)}",
        )

    def _to_status(stats: dict[str, Any]) -> CollectionStatus:
        return CollectionStatus(
            collection=stats["collection"],
            points_count=stats.get("points_count", 0) or 0,
            status=stats.get("status", "unknown"),
            exists=stats.get("exists", False),
            is_populated=bool(stats.get("points_count", 0)),
        )

    return CollectionsStatusResponse(
        code_repo=_to_status(all_stats.get("code_repo", {"collection": "code_repo", "points_count": 0, "status": "unknown", "exists": False})),
        app_docs=_to_status(all_stats.get("app_docs", {"collection": "app_docs", "points_count": 0, "status": "unknown", "exists": False})),
        incident_reports=_to_status(all_stats.get("incident_reports", {"collection": "incident_reports", "points_count": 0, "status": "unknown", "exists": False})),
    )


# ---------------------------------------------------------------------------
# Wiki API Models
# ---------------------------------------------------------------------------


class WikiModuleEntry(BaseModel):
    """A single module in the wiki tree."""
    name: str
    description: str = ""
    components: list[str] = Field(default_factory=list)
    children: dict[str, Any] = Field(default_factory=dict)
    has_doc: bool = False


class WikiTreeResponse(BaseModel):
    """Full module tree for a project wiki."""
    project_id: str
    modules: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = ""


class WikiModuleResponse(BaseModel):
    """A single module's documentation."""
    module_name: str
    content: str
    project_id: str


class WikiOverviewResponse(BaseModel):
    """Repository overview."""
    content: str
    project_id: str
    repo_name: str = ""


class WikiSearchResult(BaseModel):
    """A single search result within wiki docs."""
    module_name: str
    snippet: str
    score: float = 0.0


class WikiSearchResponse(BaseModel):
    """Response for wiki search."""
    query: str
    results: list[WikiSearchResult] = Field(default_factory=list)


class WikiRegenRequest(BaseModel):
    """Trigger wiki regeneration."""
    project_id: str
    repo_path: str
    repo_name: str | None = None
    incremental: bool = False
    changed_files: list[str] = Field(default_factory=list)
    removed_files: list[str] = Field(default_factory=list)


class WikiRegenResponse(BaseModel):
    """Response after triggering wiki regen."""
    task_id: str
    status: str = "queued"


# ---------------------------------------------------------------------------
# Wiki Routes
# ---------------------------------------------------------------------------


@app.get(
    "/wiki/{project_id}/tree",
    response_model=WikiTreeResponse,
    tags=["Wiki"],
)
async def wiki_tree(project_id: str) -> WikiTreeResponse:
    """
    Return the module tree for a project's wiki.

    Reads the module_tree.json from the wiki docs directory.
    """
    wiki_dir = os.path.join(_get_wiki_base(), project_id)
    tree_path = os.path.join(wiki_dir, "module_tree.json")

    if not os.path.exists(tree_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No wiki found for project {project_id}",
        )

    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))

    # Annotate each node with has_doc
    def _annotate(subtree: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, info in subtree.items():
            doc_path = os.path.join(wiki_dir, f"{name}.md")
            entry = {**info, "has_doc": os.path.exists(doc_path)}
            if info.get("children"):
                entry["children"] = _annotate(info["children"])
            result[name] = entry
        return result

    annotated = _annotate(tree)

    # Load metadata for generated_at timestamp
    meta_path = os.path.join(wiki_dir, "metadata.json")
    generated_at = ""
    if os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            generated_at = meta.get("completed_at", meta.get("started_at", ""))
        except (json.JSONDecodeError, OSError):
            pass

    return WikiTreeResponse(
        project_id=project_id,
        modules=annotated,
        generated_at=generated_at,
    )


@app.get(
    "/wiki/{project_id}/overview",
    response_model=WikiOverviewResponse,
    tags=["Wiki"],
)
async def wiki_overview(project_id: str) -> WikiOverviewResponse:
    """Return the repository overview document."""
    wiki_dir = os.path.join(_get_wiki_base(), project_id)
    overview_path = os.path.join(wiki_dir, "overview.md")

    if not os.path.exists(overview_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No overview found for project {project_id}",
        )

    content = Path(overview_path).read_text(encoding="utf-8")

    # Try to get repo name from metadata
    repo_name = project_id
    meta_path = os.path.join(wiki_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            repo_name = meta.get("repo_name", project_id)
        except (json.JSONDecodeError, OSError):
            pass

    return WikiOverviewResponse(
        content=content,
        project_id=project_id,
        repo_name=repo_name,
    )


@app.get(
    "/wiki/{project_id}/modules/{module_name}",
    response_model=WikiModuleResponse,
    tags=["Wiki"],
)
async def wiki_module(project_id: str, module_name: str) -> WikiModuleResponse:
    """Return the documentation for a specific module."""
    wiki_dir = os.path.join(_get_wiki_base(), project_id)
    # Sanitise module name to prevent path traversal
    safe_name = Path(module_name).name
    doc_path = os.path.join(wiki_dir, f"{safe_name}.md")

    if not os.path.exists(doc_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Module {module_name} not found in project {project_id}",
        )

    content = Path(doc_path).read_text(encoding="utf-8")
    return WikiModuleResponse(
        module_name=safe_name,
        content=content,
        project_id=project_id,
    )


@app.get(
    "/wiki/{project_id}/modules",
    tags=["Wiki"],
)
async def wiki_module_list(project_id: str) -> list[str]:
    """Return a flat list of all module names with docs."""
    wiki_dir = os.path.join(_get_wiki_base(), project_id)
    if not os.path.isdir(wiki_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No wiki found for project {project_id}",
        )

    modules: list[str] = []
    for fname in sorted(os.listdir(wiki_dir)):
        if fname.endswith(".md") and fname not in ("overview.md",):
            modules.append(fname.removesuffix(".md"))
    return modules


@app.get(
    "/wiki/{project_id}/search",
    response_model=WikiSearchResponse,
    tags=["Wiki"],
)
async def wiki_search(project_id: str, q: str = "") -> WikiSearchResponse:
    """
    Full-text keyword search across all wiki module docs.

    For semantic search use the /agent/query endpoint with project_id.
    """
    if not q or len(q.strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' must be at least 2 characters",
        )

    wiki_dir = os.path.join(_get_wiki_base(), project_id)
    if not os.path.isdir(wiki_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No wiki found for project {project_id}",
        )

    query_lower = q.strip().lower()
    results: list[WikiSearchResult] = []

    for fname in sorted(os.listdir(wiki_dir)):
        if not fname.endswith(".md"):
            continue
        module_name = fname.removesuffix(".md")
        try:
            content = Path(os.path.join(wiki_dir, fname)).read_text(encoding="utf-8")
        except OSError:
            continue

        content_lower = content.lower()
        idx = content_lower.find(query_lower)
        if idx == -1:
            continue

        # Build snippet around the first match
        start = max(0, idx - 80)
        end = min(len(content), idx + len(q) + 120)
        snippet = content[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        # Naive scoring: count occurrences
        count = content_lower.count(query_lower)
        results.append(WikiSearchResult(
            module_name=module_name,
            snippet=snippet,
            score=float(count),
        ))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)

    return WikiSearchResponse(query=q, results=results[:20])


@app.post(
    "/wiki/regenerate",
    response_model=WikiRegenResponse,
    tags=["Wiki"],
)
async def wiki_regenerate(request: WikiRegenRequest) -> WikiRegenResponse:
    """
    Trigger wiki (re)generation as a Celery background task.

    Set ``incremental: true`` and provide ``changed_files`` for
    incremental regeneration (Phase 3).
    """
    from services.docgen.tasks import generate_wiki, generate_wiki_incremental

    try:
        if request.incremental and request.changed_files:
            task = generate_wiki_incremental.apply_async(
                kwargs={
                    "project_id": request.project_id,
                    "repo_path": request.repo_path,
                    "changed_files": request.changed_files,
                    "removed_files": request.removed_files,
                    "repo_name": request.repo_name,
                },
                queue="docgen",
            )
        else:
            task = generate_wiki.apply_async(
                kwargs={
                    "project_id": request.project_id,
                    "repo_path": request.repo_path,
                    "repo_name": request.repo_name,
                },
                queue="docgen",
            )
    except Exception as exc:
        log.error("wiki_regenerate_dispatch_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to dispatch wiki generation: {str(exc)}",
        )

    return WikiRegenResponse(task_id=task.id, status="queued")


@app.get(
    "/wiki/projects",
    tags=["Wiki"],
)
async def wiki_list_projects() -> list[str]:
    """List all project IDs that have generated wikis."""
    base = _get_wiki_base()
    if not os.path.isdir(base):
        return []
    return sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
        and os.path.exists(os.path.join(base, d, "module_tree.json"))
    )


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions."""
    log.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
