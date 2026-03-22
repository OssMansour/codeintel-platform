#!/usr/bin/env bash
# =============================================================================
# CodeIntel Platform — Setup Script
# =============================================================================
# Usage: ./scripts/setup.sh
#
# This script:
#   1. Checks prerequisites (Docker, Docker Compose, Ollama)
#   2. Creates required data directories
#   3. Copies .env.example to .env (if not already present)
#   4. Pulls required Ollama models
#   5. Starts all services via docker-compose
#   6. Runs health checks on all services
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Script Directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     CodeIntel Platform — Setup                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check Prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

# Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Install it from https://docs.docker.com/get-docker/"
fi
DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
success "Docker $DOCKER_VERSION found"

# Docker Compose
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    error "Docker Compose is not installed. Install it from https://docs.docker.com/compose/install/"
fi
success "Docker Compose found (using: $COMPOSE_CMD)"

# Ollama (optional for docker setup, but recommended for testing)
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    success "Ollama found: $OLLAMA_VERSION"
    OLLAMA_AVAILABLE=true
else
    warn "Ollama not found locally. Ollama will run inside Docker (slower first start)."
    warn "For better performance, install Ollama: https://ollama.ai"
    OLLAMA_AVAILABLE=false
fi

# Python (for initial_index.py)
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    success "Python found: $PY_VERSION"
else
    warn "Python 3 not found. You'll need it to run initial_index.py outside Docker."
fi

echo ""

# ---------------------------------------------------------------------------
# Step 2: Create Required Directories
# ---------------------------------------------------------------------------
info "Creating data directories..."

DATA_DIRS=(
    "/data/repos"
    "/data/indexes/embed_cache"
    "/data/models/embed"
    "/data/models/reranker"
    "/data/models/ollama"
    "/data/sources"
    "/data/qdrant"
)

for dir in "${DATA_DIRS[@]}"; do
    if mkdir -p "$dir" 2>/dev/null; then
        success "Created $dir"
    else
        warn "Could not create $dir (may need sudo or different path)"
        warn "Run: sudo mkdir -p $dir && sudo chown -R \$(whoami) $dir"
    fi
done

echo ""

# ---------------------------------------------------------------------------
# Step 3: Environment Configuration
# ---------------------------------------------------------------------------
info "Setting up environment configuration..."

if [ -f "$PROJECT_ROOT/.env" ]; then
    warn ".env already exists — keeping existing configuration"
    warn "To reset: rm .env && ./scripts/setup.sh"
else
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    success "Created .env from .env.example"
    echo ""
    echo -e "${YELLOW}════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  ACTION REQUIRED: Edit .env before continuing${NC}"
    echo -e "${YELLOW}════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Minimum required changes:"
    echo "  - GITLAB_URL=https://your-gitlab.internal.com"
    echo "  - GITLAB_TOKEN=glpat-XXXXXXXXXX"
    echo "  - GITLAB_WEBHOOK_SECRET=<generate a random secret>"
    echo "  - SECRET_KEY=<generate with: openssl rand -hex 32>"
    echo ""
    echo "  Optional (if you have incident reports):"
    echo "  - INCIDENT_REPORTS_PATH=/data/sources/incident_reports.md"
    echo ""
    read -p "Press Enter after editing .env to continue, or Ctrl+C to stop..."
fi

echo ""

# ---------------------------------------------------------------------------
# Step 4: Pull Ollama Models
# ---------------------------------------------------------------------------
info "Pulling required Ollama models..."
echo "  This may take a while on first run (models are several GB each)"
echo ""

# Load model names from .env
source "$PROJECT_ROOT/.env" 2>/dev/null || true

PRIMARY_MODEL="${OLLAMA_PRIMARY_MODEL:-qwen2.5-coder:14b-instruct-q5_K_M}"
FALLBACK_MODEL="${OLLAMA_FALLBACK_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"
EMBED_MODEL_OLLAMA="nomic-embed-text"  # Ollama version for testing

if [ "$OLLAMA_AVAILABLE" = "true" ]; then
    # Pull via local Ollama
    info "Pulling primary model: $PRIMARY_MODEL"
    ollama pull "$PRIMARY_MODEL" && success "Primary model ready" || warn "Failed to pull primary model — will retry via Docker"

    info "Pulling fallback model: $FALLBACK_MODEL"
    ollama pull "$FALLBACK_MODEL" && success "Fallback model ready" || warn "Failed to pull fallback model"

    info "Pulling nomic-embed-text (for testing)"
    ollama pull "$EMBED_MODEL_OLLAMA" && success "Embed model ready" || warn "Failed to pull embed model"
else
    warn "Skipping Ollama model pull (Ollama not available locally)"
    warn "Models will be pulled when the Ollama Docker container starts"
    warn "This will happen automatically but may take 10-30 minutes"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 5: Place Application Docs
# ---------------------------------------------------------------------------
info "Checking for required source documents..."

APP_DOCS_PATH="${APP_DOCS_PATH:-/data/sources/Application_documentation.md}"
if [ -f "$APP_DOCS_PATH" ]; then
    success "Application_documentation.md found at $APP_DOCS_PATH"
else
    warn "Application_documentation.md not found at $APP_DOCS_PATH"
    warn "Creating a placeholder — replace with your actual documentation"
    cat > "$APP_DOCS_PATH" << 'EOF'
# Application Documentation

This is a placeholder for your application documentation.
Replace this file with your actual Application_documentation.md.

## Overview
Describe your application here.

## Features
List your application features here.

## API Reference
Document your API here.
EOF
    success "Placeholder created at $APP_DOCS_PATH"
fi

INCIDENT_PATH="${INCIDENT_REPORTS_PATH:-/data/sources/incident_reports.md}"
if [ -f "$INCIDENT_PATH" ]; then
    success "incident_reports.md found at $INCIDENT_PATH"
else
    info "incident_reports.md not found at $INCIDENT_PATH (optional — OK to skip)"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 6: Start Docker Services
# ---------------------------------------------------------------------------
info "Starting Docker Compose services..."
echo ""

$COMPOSE_CMD -f "$PROJECT_ROOT/docker-compose.yml" up -d --build

echo ""
info "Waiting for services to become healthy..."
sleep 10

# ---------------------------------------------------------------------------
# Step 7: Health Checks
# ---------------------------------------------------------------------------
echo ""
info "Running health checks..."

HEALTH_OK=true

check_health() {
    local service_name="$1"
    local url="$2"
    local max_attempts="${3:-30}"
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
        if [ $((attempt % 5)) -eq 0 ]; then
            info "Still waiting for $service_name... ($attempt/$max_attempts)"
        fi
    done

    warn "$service_name health check failed after ${max_attempts} attempts"
    HEALTH_OK=false
    return 1
}

check_health "Qdrant"    "http://localhost:6333/healthz"    30
check_health "Redis"     "http://localhost:6379"             10  || true  # Redis doesn't have HTTP health
check_health "Ollama"    "http://localhost:11434/api/tags"  60
check_health "Ingestion" "http://localhost:8000/health"     30
check_health "Agent"     "http://localhost:8001/health"     30
check_health "Frontend"  "http://localhost:3000"            30

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "╔══════════════════════════════════════════════════════╗"
echo "║     CodeIntel Platform — Setup Summary                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

if [ "$HEALTH_OK" = "true" ]; then
    echo -e "${GREEN}All services are running!${NC}"
else
    echo -e "${YELLOW}Some services may not be fully ready yet.${NC}"
    echo "  Run: docker compose logs <service-name> to debug"
fi

echo ""
echo "Service URLs:"
echo "  Ingestion API:  http://localhost:8000/docs"
echo "  Agent API:      http://localhost:8001/docs"
echo "  Frontend:       http://localhost:3000"
echo "  Qdrant UI:      http://localhost:6333/dashboard"
echo ""
echo "Next steps:"
echo "  1. Run initial indexing:"
echo "     python scripts/initial_index.py \\"
echo "       --project-id your-project \\"
echo "       --repo-path /data/repos/your-project \\"
echo "       --app-docs /data/sources/Application_documentation.md"
echo ""
echo "  2. Configure GitLab webhook:"
echo "     URL: http://your-server:8000/webhook"
echo "     Secret: \${GITLAB_WEBHOOK_SECRET} from .env"
echo "     Events: Push events, Merge request events, Issue events"
echo ""
echo "  3. Test a query:"
echo '     curl -X POST http://localhost:8001/agent/query \'
echo '       -H "Content-Type: application/json" \'
echo '       -d "{\"query\": \"How does the authentication work?\", \"project_id\": \"your-project\"}"'
echo ""
