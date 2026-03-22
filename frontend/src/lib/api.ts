/**
 * CodeIntel Frontend — API Client
 *
 * Typed client for the Agent API wiki endpoints + agent query.
 */

const API_BASE = process.env.NEXT_PUBLIC_AGENT_API_URL || "http://localhost:8001";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface WikiModuleTree {
  project_id: string;
  modules: Record<string, WikiModuleNode>;
  generated_at: string;
}

export interface WikiModuleNode {
  path: string;
  components: string[];
  description: string;
  children: Record<string, WikiModuleNode>;
  has_doc: boolean;
}

export interface WikiModuleDoc {
  module_name: string;
  content: string;
  project_id: string;
}

export interface WikiOverview {
  content: string;
  project_id: string;
  repo_name: string;
}

export interface WikiSearchResult {
  module_name: string;
  snippet: string;
  score: number;
}

export interface WikiSearchResponse {
  query: string;
  results: WikiSearchResult[];
}

export interface ChatSource {
  file: string;
  symbol: string;
  start_line: number;
  end_line: number;
  collection: string;
  permalink: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  confidence: number;
  collection_hits: Record<string, number>;
  tool_calls_made: string[];
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Wiki endpoints
// ---------------------------------------------------------------------------

export async function fetchProjects(): Promise<string[]> {
  return get<string[]>("/wiki/projects");
}

export async function fetchModuleTree(projectId: string): Promise<WikiModuleTree> {
  return get<WikiModuleTree>(`/wiki/${encodeURIComponent(projectId)}/tree`);
}

export async function fetchOverview(projectId: string): Promise<WikiOverview> {
  return get<WikiOverview>(`/wiki/${encodeURIComponent(projectId)}/overview`);
}

export async function fetchModuleDoc(
  projectId: string,
  moduleName: string
): Promise<WikiModuleDoc> {
  return get<WikiModuleDoc>(
    `/wiki/${encodeURIComponent(projectId)}/modules/${encodeURIComponent(moduleName)}`
  );
}

export async function fetchModuleList(projectId: string): Promise<string[]> {
  return get<string[]>(`/wiki/${encodeURIComponent(projectId)}/modules`);
}

export async function searchWiki(
  projectId: string,
  query: string
): Promise<WikiSearchResponse> {
  return get<WikiSearchResponse>(
    `/wiki/${encodeURIComponent(projectId)}/search?q=${encodeURIComponent(query)}`
  );
}

// ---------------------------------------------------------------------------
// Agent chat
// ---------------------------------------------------------------------------

export async function agentQuery(
  query: string,
  projectId: string = ""
): Promise<ChatResponse> {
  return post<ChatResponse>("/agent/query", {
    query,
    project_id: projectId,
    stream: false,
  });
}

/**
 * Streaming agent query — yields SSE events.
 */
export async function* agentQueryStream(
  query: string,
  projectId: string = ""
): AsyncGenerator<{ type: string; content: unknown }> {
  const res = await fetch(`${API_BASE}/agent/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, project_id: projectId, stream: true }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          // ignore parse errors
        }
      }
    }
  }
}
