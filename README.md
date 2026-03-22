# CodeIntel Platform

An **on-premises AI Code Intelligence Platform** that provides semantic code search, hierarchical localization, multi-source context fusion, and automated documentation generation — entirely CPU-based with no cloud API calls.

Supports **GitLab** and **GitHub** repositories out of the box via a unified SCM abstraction layer.

---

## Architecture Overview

```
GitLab / GitHub Repos     Application_docs.md         incident_reports.md
     │                           │                            │
     ▼                           ▼                            ▼
 Ingestion                  Doc Indexer                  Doc Indexer
 (FastAPI + HMAC)           (Markdown chunker)           (optional)
     │                           │                            │
     └──────────┬────────────────┘────────────────────────────┘
                │
                ▼
           tree-sitter Parser (12 languages)
                │
                ▼
           Semantic Chunker → Stable Chunk IDs (SHA-256)
                │
                ▼
       CodeRankEmbed (768-dim, CPU) → File-based embedding cache
                │
                ▼
         Qdrant (3 collections: code_repo | app_docs | incident_reports)
                │
                ▼
        LangGraph ReAct Agent (6 tools, SQLite checkpointing)
          ├── search_code         → code_repo hybrid search
          ├── search_app_docs     → app_docs hybrid search
          ├── search_incidents    → incident_reports (graceful [] if empty)
          ├── keyword_search      → BM25 exact identifier search
          ├── traverse_graph      → dependency graph traversal
          └── retrieve_entity     → exact source + SCM permalink
                │
                ▼
       Cross-encoder Reranker (ms-marco-MiniLM-L6-v2, 34MB)
                │
                ▼
       Ollama LLM (Qwen2.5-Coder-14B Q5_K_M, CPU)
                │
                ▼
         FastAPI Agent API + Next.js Frontend
```

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-SCM Support** | GitLab and GitHub via a unified `SCMClient` interface — switch with one env var |
| **12-Language Parsing** | Python, JavaScript, TypeScript, Go, Java, C, C++, C#, Kotlin, PHP + fallback chunker |
| **Hybrid Search** | Dense vector (CodeRankEmbed) + BM25 lexical retrieval for best recall |
| **ReAct Agent** | LangGraph-based reasoning agent with 6 specialized tools |
| **Cross-Encoder Reranking** | ms-marco-MiniLM-L6-v2 for precision re-scoring |
| **Automated Wiki** | Hierarchical doc generation with incremental refresh on git push |
| **Benchmarking Suite** | 4 quality / performance suites with threshold-gated CI checks |
| **100% On-Prem** | CPU-only, no cloud API calls — runs air-gapped on 32GB+ RAM |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker | ≥ 24.0 | Required |
| Docker Compose | ≥ 2.0 | Required |
| Ollama | ≥ 0.3.0 | Optional (can run in Docker) |
| Python | ≥ 3.11 | For `initial_index.py` |
| RAM | ≥ 32 GB | For 14B model on CPU |
| Disk | ≥ 100 GB | Models + indexes + repos |
| CPU | ≥ 16 cores | For acceptable inference speed |

---

## Project Structure

```
codeintel-platform/
├── .env.example                 # Environment variable template
├── docker-compose.yml           # All 10 services + volumes
├── Dockerfile.python            # Shared Python service image
├── requirements.txt             # Python dependencies
│
├── services/                    # ── Backend Python packages ──
│   ├── __init__.py
│   ├── ingestion/               # SCM webhooks + indexing triggers
│   │   ├── main.py              #   FastAPI app (GitLab + GitHub endpoints)
│   │   ├── scm_provider.py      #   Unified SCMClient ABC + shared utilities
│   │   ├── gitlab_client.py     #   GitLab implementation (python-gitlab)
│   │   └── github_client.py     #   GitHub implementation (PyGithub)
│   ├── parsing/                 # Code analysis
│   │   ├── parser.py            #   tree-sitter multi-language parser
│   │   └── chunker.py           #   Symbol-boundary semantic chunker
│   ├── indexing/                # Vector storage
│   │   ├── embedder.py          #   CodeRankEmbed with file-based cache
│   │   ├── qdrant_store.py      #   Qdrant client (3 collections)
│   │   └── doc_indexer.py       #   Markdown document indexer
│   ├── agent/                   # AI reasoning
│   │   ├── agent.py             #   LangGraph ReAct state machine
│   │   ├── api.py               #   FastAPI Agent API + SSE streaming
│   │   ├── tools.py             #   6 agent tools
│   │   └── reranker.py          #   Cross-encoder reranker
│   ├── docgen/                  # Documentation generation
│   │   ├── tasks.py             #   Celery task definitions
│   │   ├── doc_generator.py     #   LLM-based doc writer
│   │   ├── wiki_generator.py    #   Hierarchical wiki builder
│   │   ├── async_wiki_generator.py
│   │   ├── incremental_generator.py
│   │   ├── doc_refiner.py       #   Iterative quality refinement
│   │   ├── module_cluster.py    #   Module grouping logic
│   │   ├── mermaid.py           #   Diagram generation
│   │   └── prompts.py           #   LLM prompt templates
│   └── llm/                     # LLM abstraction
│       ├── config.py            #   Provider settings
│       ├── base.py              #   LLMProvider ABC
│       ├── factory.py           #   Provider factory
│       └── providers/           #   Ollama, llama.cpp, Azure, Bedrock
│
├── frontend/                    # ── Next.js UI ──
│   ├── Dockerfile.frontend      #   Multi-stage Node build
│   ├── package.json
│   └── src/
│       ├── app/                 #   Pages (chat + wiki browser)
│       ├── components/          #   ChatSidebar, ModuleTree, SearchBar, MarkdownRenderer
│       └── lib/api.ts           #   Agent API client
│
├── benchmarks/                  # ── Quality & Performance Suite ──
│   ├── cli.py                   #   CLI entry point
│   ├── retrieval.py             #   Retrieval quality (MRR, Recall@K)
│   ├── latency.py               #   Latency percentile profiling
│   ├── wiki_quality.py          #   Wiki generation quality scoring
│   ├── regression.py            #   Answer regression detection
│   ├── report.py                #   HTML/JSON report generator
│   └── config.py                #   Threshold configuration
│
└── scripts/
    ├── setup.sh                 #   One-command environment bootstrap
    └── initial_index.py         #   First-run repository indexing
```

---

## Quick Start

### Step 1 — Clone and configure

```bash
git clone <this-repo>
cd codeintel-platform
cp .env.example .env
```

Edit `.env` with your SCM credentials. **Choose one provider:**

<details>
<summary><strong>GitLab (default)</strong></summary>

```bash
SCM_PROVIDER=gitlab
GITLAB_URL=https://your-gitlab.internal.com
GITLAB_TOKEN=glpat-XXXXXXXXXX
GITLAB_WEBHOOK_SECRET=$(openssl rand -hex 20)
GITLAB_PROJECT_ID=group/repo-name
SECRET_KEY=$(openssl rand -hex 32)
```
</details>

<details>
<summary><strong>GitHub</strong></summary>

```bash
SCM_PROVIDER=github
GITHUB_URL=https://github.com
GITHUB_API_URL=https://api.github.com
GITHUB_TOKEN=ghp_XXXXXXXXXX
GITHUB_REPO=owner/repo-name
GITHUB_WEBHOOK_SECRET=$(openssl rand -hex 20)
SECRET_KEY=$(openssl rand -hex 32)
```
</details>

### Step 2 — Run setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Create data directories (`/data/repos`, `/data/indexes`, `/data/models`)
- Pull Ollama models (14B Q5_K_M, 7B Q4_K_M)
- Start all Docker services
- Run health checks

### Step 3 — Clone your repository locally

```bash
# GitLab
git clone https://your-gitlab.internal.com/group/my-repo /data/repos/group_my-repo

# GitHub
git clone https://github.com/owner/my-repo /data/repos/owner_my-repo
```

### Step 4 — Run initial indexing

```bash
python scripts/initial_index.py \
    --project-id group/my-repo \
    --repo-path /data/repos/group_my-repo \
    --app-docs /data/sources/Application_documentation.md \
    --incident-reports /data/sources/incident_reports.md
```

This will index your code, application docs, and (if available) incident reports into Qdrant.

### Step 5 — Configure SCM webhook

<details>
<summary><strong>GitLab</strong></summary>

**Settings → Webhooks → Add new webhook**

| Field | Value |
|---|---|
| URL | `http://your-server:8000/webhook` |
| Secret token | `${GITLAB_WEBHOOK_SECRET}` from `.env` |
| Trigger | Push events, Merge request events, Issues events |
</details>

<details>
<summary><strong>GitHub</strong></summary>

**Settings → Webhooks → Add webhook**

| Field | Value |
|---|---|
| Payload URL | `http://your-server:8000/webhook/github` |
| Content type | `application/json` |
| Secret | `${GITHUB_WEBHOOK_SECRET}` from `.env` |
| Events | Pushes, Pull requests, Issues |
</details>

---

## Configuration Reference

### Core Settings (.env)

#### SCM Provider

| Variable | Description | Default |
|---|---|---|
| `SCM_PROVIDER` | Active SCM platform (`gitlab` or `github`) | `gitlab` |

#### GitLab

| Variable | Description | Example |
|---|---|---|
| `GITLAB_URL` | GitLab instance URL | `https://gitlab.internal.com` |
| `GITLAB_TOKEN` | Personal access token (`api`, `read_repository`) | `glpat-XXXX` |
| `GITLAB_WEBHOOK_SECRET` | Webhook HMAC secret | `random-32-chars` |
| `GITLAB_PROJECT_ID` | Default project | `group/repo` |

#### GitHub

| Variable | Description | Example |
|---|---|---|
| `GITHUB_URL` | GitHub web URL | `https://github.com` |
| `GITHUB_API_URL` | GitHub API base URL | `https://api.github.com` |
| `GITHUB_TOKEN` | Personal access token (`repo` scope) | `ghp_XXXX` |
| `GITHUB_REPO` | Default repository (`owner/repo`) | `myorg/backend` |
| `GITHUB_WEBHOOK_SECRET` | Webhook HMAC-SHA256 secret | `random-32-chars` |

#### Ollama (LLM)

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_PRIMARY_MODEL` | Primary LLM | `qwen2.5-coder:14b-instruct-q5_K_M` |
| `OLLAMA_FALLBACK_MODEL` | Fallback LLM | `qwen2.5-coder:7b-instruct-q4_K_M` |
| `OLLAMA_BATCH_MODEL` | Batch doc-gen LLM | `qwen2.5-coder:32b-instruct-q4_K_M` |
| `OLLAMA_NUM_CTX` | Context window (tokens) | `16384` |
| `OLLAMA_NUM_THREADS` | CPU threads | `16` |

#### Qdrant

| Variable | Description | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Auth key (empty = no auth) | `` |
| `QDRANT_COLLECTION_CODE` | Code collection name | `code_repo` |
| `QDRANT_COLLECTION_APP_DOCS` | Docs collection name | `app_docs` |
| `QDRANT_COLLECTION_INCIDENTS` | Incidents collection name | `incident_reports` |

#### Embeddings

| Variable | Description | Default |
|---|---|---|
| `EMBED_MODEL` | HuggingFace model ID | `nomic-ai/CodeRankEmbed` |
| `EMBED_DEVICE` | Inference device | `cpu` |
| `EMBED_BATCH_SIZE` | Texts per batch | `32` |

#### Benchmarks

| Variable | Description | Default |
|---|---|---|
| `BENCH_AGENT_API_URL` | Agent API for benchmark calls | `http://agent:8001` |
| `BENCH_OUTPUT_DIR` | Report output directory | `/data/benchmarks/results` |

---

## Vector Store Sources

The platform indexes 3 distinct knowledge sources, each in its own Qdrant collection:

### `code_repo` — Always Available
- **Source**: GitLab / GitHub repository code
- **Content**: Every function, class, and method parsed by tree-sitter (12 languages)
- **Updated**: Automatically via SCM webhooks on every push
- **Languages**: Python, JavaScript, TypeScript, Go, Java, C, C++, C#, Kotlin, PHP

### `app_docs` — Always Available
- **Source**: `Application_documentation.md` + AI-generated module docs
- **Content**: Product documentation chunked by Markdown heading hierarchy
- **Updated**: On startup if empty; manually via `POST /index/static-docs`

### `incident_reports` — Conditionally Available
- **Source**: `incident_reports.md` (if provided)
- **Content**: Post-mortems, root cause analyses, remediation records
- **Updated**: Manually via `POST /index/static-docs`
- **Graceful degradation**: Returns `[]` if file not provided — no errors raised

---

## API Reference

### Ingestion Service (port 8000)

#### `GET /health`
Health check endpoint.

```bash
curl http://localhost:8000/health
# {"status": "healthy", "service": "ingestion", "version": "1.1.0"}
```

#### `POST /webhook`
Receive GitLab webhook events (HMAC verified via `X-Gitlab-Token`).

```bash
curl -X POST http://localhost:8000/webhook \
  -H "X-Gitlab-Token: your-webhook-secret" \
  -H "Content-Type: application/json" \
  -d '{"object_kind": "push", "project": {"path_with_namespace": "group/repo"}, "commits": []}'
```

#### `POST /webhook/github`
Receive GitHub webhook events (HMAC-SHA256 verified via `X-Hub-Signature-256`).

```bash
curl -X POST http://localhost:8000/webhook/github \
  -H "X-GitHub-Event: push" \
  -H "X-Hub-Signature-256: sha256=..." \
  -H "Content-Type: application/json" \
  -d '{"ref": "refs/heads/main", "repository": {"full_name": "owner/repo"}, "commits": []}'
```

#### `POST /index/full`
Trigger full repository re-index.

```bash
curl -X POST http://localhost:8000/index/full \
  -H "Content-Type: application/json" \
  -d '{"project_id": "group/my-repo", "repo_path": "/data/repos/group_my-repo"}'
```

#### `POST /index/file`
Re-index a single file.

```bash
curl -X POST http://localhost:8000/index/file \
  -H "Content-Type: application/json" \
  -d '{"project_id": "group/my-repo", "file_path": "src/payment/retry.py", "repo_path": "/data/repos/group_my-repo"}'
```

#### `POST /index/static-docs`
Re-index Application_documentation.md and incident_reports.md.

```bash
curl -X POST http://localhost:8000/index/static-docs
```

---

### Agent Service (port 8001)

#### `POST /agent/query`
Natural language code intelligence query.

```bash
# Standard query
curl -X POST http://localhost:8001/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does the payment retry logic work?",
    "project_id": "group/payments-service"
  }'

# Streaming query (SSE)
curl -X POST http://localhost:8001/agent/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "What happens when a database connection fails?",
    "project_id": "group/my-repo",
    "stream": true
  }'
```

**Response:**
```json
{
  "answer": "The payment retry logic is implemented in `payment/retry.py`...",
  "sources": [
    {
      "file": "payment/retry.py",
      "symbol": "retry_payment",
      "start_line": 45,
      "end_line": 78,
      "collection": "code_repo",
      "permalink": "https://github.com/myorg/payments/blob/abc123/payment/retry.py#L45-L78"
    }
  ],
  "confidence": 0.92,
  "collection_hits": {"code_repo": 5, "app_docs": 2, "incident_reports": 1}
}
```

#### `POST /agent/localize`
Localize an issue to specific code lines (on-call engineer endpoint).

```bash
curl -X POST http://localhost:8001/agent/localize \
  -H "Content-Type: application/json" \
  -d '{
    "issue_text": "TypeError: NoneType has no attribute retry_count in payment processing",
    "project_id": "group/payments-service",
    "severity": "P1"
  }'
```

#### `GET /agent/collections/status`
Check collection health.

```bash
curl http://localhost:8001/agent/collections/status
```

**Response:**
```json
{
  "code_repo": {"collection": "code_repo", "points_count": 42381, "status": "green", "exists": true, "is_populated": true},
  "app_docs": {"collection": "app_docs", "points_count": 847, "status": "green", "exists": true, "is_populated": true},
  "incident_reports": {"collection": "incident_reports", "points_count": 0, "status": "green", "exists": true, "is_populated": false}
}
```

---

## Benchmarking

Run the full benchmark suite (retrieval quality, latency profiling, wiki quality, answer regression):

```bash
# Via Docker Compose (recommended)
docker compose run --rm benchmarks run --suite all --check-thresholds

# Individual suites
docker compose run --rm benchmarks run --suite retrieval
docker compose run --rm benchmarks run --suite latency
docker compose run --rm benchmarks run --suite wiki
docker compose run --rm benchmarks run --suite regression

# Generate HTML report only
docker compose run --rm benchmarks report
```

Reports are written to `/data/benchmarks/results/`.

---

## Service Architecture

| Service | Port | Technology | Purpose |
|---|---|---|---|
| `ingestion` | 8000 | FastAPI + Gunicorn | Webhook receiver (GitLab + GitHub) + index triggers |
| `agent` | 8001 | FastAPI + Gunicorn | LangGraph ReAct agent API + SSE streaming |
| `indexer` | — | Celery worker | Async code indexing jobs |
| `docgen` | — | Celery worker | Doc generation + wiki refresh jobs |
| `celery-beat` | — | Celery beat | Scheduled tasks (daily static-doc refresh) |
| `qdrant` | 6333 | Qdrant | Vector database (3 collections) |
| `redis` | 6379 | Redis | Celery broker + result backend |
| `ollama` | 11434 | Ollama | Local LLM server (CPU) |
| `frontend` | 3000 | Next.js | Chat interface + wiki browser |
| `benchmarks` | — | Python CLI | On-demand quality & perf evaluation |

---

## SCM Abstraction Layer

The platform uses a unified `SCMClient` interface so all downstream code (chunker, webhook handlers, agent tools) is SCM-agnostic:

```
SCMClient (ABC)                  ← services/ingestion/scm_provider.py
├── GitLabClient(SCMClient)      ← services/ingestion/gitlab_client.py
└── GitHubClient(SCMClient)      ← services/ingestion/github_client.py
```

**Switch providers** by setting `SCM_PROVIDER=github` in `.env` — no code changes required.

Shared data classes (`DiffFile`, `CommitInfo`, `PRInfo`, `IssueInfo`, `CommentInfo`) and utilities (`build_permalink`, `git_clone_or_pull`) live in `scm_provider.py` to avoid duplication.

---

## LLM Provider Abstraction

The platform supports 4 LLM backends via a pluggable provider system:

| Provider | Module | Use Case |
|---|---|---|
| **Ollama** (default) | `services/llm/providers/ollama.py` | On-prem CPU inference |
| **llama.cpp** | `services/llm/providers/llamacpp.py` | Direct GGUF via OpenAI-compatible API |
| **Azure OpenAI** | `services/llm/providers/azure.py` | Cloud deployment |
| **Amazon Bedrock** | `services/llm/providers/bedrock.py` | AWS deployment |

Set `LLM_PROVIDER=ollama|llamacpp|azure|bedrock` in `.env`.

---

## Contributing

### Code Style
- Python 3.11+ with type hints throughout
- `pydantic-settings` for all configuration
- `structlog` for structured logging
- No hardcoded secrets — `.env` only
- Chunk IDs must be deterministic: `sha256("{project_id}:{file_path}:{symbol_name}:{content}")`

### Adding a New Language
1. Add extension mapping in `services/parsing/parser.py` → `EXTENSION_TO_LANGUAGE`
2. Add tree-sitter query in `services/parsing/parser.py` → `QUERIES`
3. Update `LANGUAGE_EXTENSIONS` in `services/docgen/tasks.py`
4. Install the corresponding `tree-sitter-{language}` package

### Adding a New SCM Provider
1. Create `services/ingestion/{provider}_client.py` implementing `SCMClient`
2. Add provider value to `SCMProvider` enum in `scm_provider.py`
3. Register in `get_scm_client()` factory
4. Add webhook endpoint in `services/ingestion/main.py`
5. Add env vars in `.env.example` and `docker-compose.yml`

### Adding a New Knowledge Source
1. Create a Qdrant collection in `services/indexing/qdrant_store.py`
2. Add an indexer in `services/indexing/doc_indexer.py`
3. Add a tool in `services/agent/tools.py`
4. Update the agent system prompt in `services/agent/agent.py`
5. Handle graceful degradation (empty collection → `[]`)

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Ollama service slow to start | Large model download | Wait 10-30 min on first run |
| GitLab webhook returns 401 | Wrong `GITLAB_WEBHOOK_SECRET` | Verify secret matches GitLab config |
| GitHub webhook returns 401 | Wrong `GITHUB_WEBHOOK_SECRET` | Check HMAC-SHA256 secret in GitHub settings |
| `code_repo` collection empty | `initial_index.py` not run | Run Step 4 of Quick Start |
| Agent returns hallucinations | Model too small or bad retrieval | Try 14B model; check embedder health |
| Out of RAM | 14B model too large | Reduce `OLLAMA_NUM_CTX` or use 7B fallback |
| Embedding very slow | CPU only, large batch | Reduce `EMBED_BATCH_SIZE` |
| Benchmark thresholds fail | Quality regression | Check `benchmarks/data/manifest.json` thresholds |

---

## License

Internal use only. See your organization's software license policy.
