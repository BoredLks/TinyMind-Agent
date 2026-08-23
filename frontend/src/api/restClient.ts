// REST client. Dev talks to the backend on :8000; the packaged app is same-origin.
const API = import.meta.env.DEV ? 'http://127.0.0.1:8000' : ''

async function http<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const payload = await res.json()
      if (payload?.detail) message = String(payload.detail)
    } catch {
      // Keep the status fallback when the server does not return JSON.
    }
    throw new Error(message)
  }
  if (res.status === 204) return null as T
  return (await res.json()) as T
}

async function httpText(path: string): Promise<{ text: string; filename: string }> {
  const res = await fetch(API + path)
  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const payload = await res.json()
      if (payload?.detail) message = String(payload.detail)
    } catch {
      // Keep the status fallback when the server does not return JSON.
    }
    throw new Error(message)
  }
  const disposition = res.headers.get('content-disposition') ?? ''
  const match = /filename="?([^";]+)"?/i.exec(disposition)
  return { text: await res.text(), filename: match?.[1] ?? 'session-export.txt' }
}

export interface SessionMeta {
  id: string
  title: string
  updated_at: string
  workflow_stage?: string
  meta_json?: string | null
  project_id?: string | null
}

export interface MessageRow {
  id: string
  role: string
  content: string
}

export interface SkillRow {
  name: string
  description: string
  source: string
  enabled: boolean
}

export interface ProjectRow {
  id: string
  name: string
  path: string
  current: boolean
  updated_at: string
  last_opened_at?: string | null
}

export interface ProviderRow {
  id: string
  label: string
  base_url: string
  model: string
  enabled: boolean
  has_api_key: boolean
}

export interface ProvidersResponse {
  active_provider_id: string
  providers: ProviderRow[]
}

export interface MemoryRow {
  id: string
  theme: string
  content: string
  created_at: string
  updated_at: string
}

export interface ProjectsResponse {
  current_project_id: string | null
  projects: ProjectRow[]
}

export const api = {
  listSessions: () => http<SessionMeta[]>('/api/sessions'),
  createSession: (title = '新会话', projectId?: string | null) =>
    http<SessionMeta>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ title, project_id: projectId ?? null }),
    }),
  renameSession: (id: string, title: string) =>
    http<SessionMeta>(`/api/sessions/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteSession: (id: string) => http(`/api/sessions/${id}`, { method: 'DELETE' }),
  getMessages: (id: string) => http<MessageRow[]>(`/api/sessions/${id}/messages`),
  undoSession: (id: string, rounds = 1) =>
    http<{ ok: boolean; deleted: number; messages: MessageRow[] }>(`/api/sessions/${id}/undo`, {
      method: 'POST',
      body: JSON.stringify({ rounds }),
    }),
  exportUrl: (id: string, format: 'md' | 'json') => `${API}/api/sessions/${id}/export?format=${format}`,
  exportSession: (id: string, format: 'md' | 'json') =>
    httpText(`/api/sessions/${id}/export?format=${format}`),
  saveSessionExport: (id: string, format: 'md' | 'json') =>
    http<{ ok: boolean; path: string; filename: string }>(
      `/api/sessions/${id}/export-file?format=${format}`,
      { method: 'POST' },
    ),

  getSettings: () => http<Record<string, unknown>>('/api/settings'),
  updateSettings: (patch: Record<string, unknown>) =>
    http<Record<string, unknown>>('/api/settings', { method: 'PUT', body: JSON.stringify(patch) }),
  setApiKey: (api_key: string) =>
    http('/api/settings/api-key', { method: 'PUT', body: JSON.stringify({ api_key }) }),
  clearApiKey: () => http('/api/settings/api-key', { method: 'DELETE' }),

  listProviders: () => http<ProvidersResponse>('/api/providers'),
  createProvider: (body: {
    id: string
    label: string
    base_url: string
    model: string
    api_key?: string
    enabled?: boolean
  }) => http<ProvidersResponse>('/api/providers', { method: 'POST', body: JSON.stringify(body) }),
  updateProvider: (id: string, body: Record<string, unknown>) =>
    http<ProvidersResponse>(`/api/providers/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  deleteProvider: (id: string) => http<ProvidersResponse>(`/api/providers/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  activateProvider: (id: string) =>
    http<ProvidersResponse>(`/api/providers/${encodeURIComponent(id)}/active`, { method: 'PUT' }),
  testProvider: (body: { id?: string; base_url?: string; api_key?: string; model?: string }) =>
    http<{ ok: boolean; latency_ms: number; detail: string }>('/api/providers/test', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  discoverProviderModels: (body: { id?: string; base_url?: string; api_key?: string }) =>
    http<{ ok: boolean; models: string[]; latency_ms: number; detail: string }>('/api/providers/discover-models', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  listMemories: () => http<{ memories: MemoryRow[]; narrative: string }>('/api/memories'),
  createMemory: (body: { theme: string; content: string }) =>
    http<MemoryRow>('/api/memories', { method: 'POST', body: JSON.stringify(body) }),
  updateMemory: (id: string, body: { theme?: string; content?: string }) =>
    http<MemoryRow>(`/api/memories/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  deleteMemory: (id: string) => http(`/api/memories/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  clearMemories: () => http('/api/memories', { method: 'DELETE' }),

  listProjects: () => http<ProjectsResponse>('/api/projects'),
  pickFolder: () =>
    http<{ available: boolean; path: string | null; error?: string }>('/api/projects/pick-folder', {
      method: 'POST',
    }),
  createProject: (path: string, name?: string) =>
    http<ProjectRow>('/api/projects', {
      method: 'POST',
      body: JSON.stringify({ path, name }),
    }),
  selectProject: (id: string) =>
    http<ProjectRow>(`/api/projects/${id}/select`, { method: 'PUT' }),
  renameProject: (id: string, name: string) =>
    http<ProjectRow>(`/api/projects/${id}`, { method: 'PATCH', body: JSON.stringify({ name }) }),
  deleteProject: (id: string) => http(`/api/projects/${id}`, { method: 'DELETE' }),

  listTools: () => http<{ tools: { name: string; description: string; requires_approval: boolean; has_doc: boolean }[]; loaded_plugins: string[] }>('/api/tools'),
  getToolDoc: (name: string) => http<{ name: string; doc: string }>(`/api/tools/${encodeURIComponent(name)}/doc`),
  listMentions: (q: string = '') => http<{ items: { type: string; name: string; label: string; description: string }[] }>(`/api/tools/mentions?q=${encodeURIComponent(q)}`),

  listSkills: () => http<SkillRow[]>('/api/skills'),
  getSkill: (name: string) =>
    http<SkillRow & { body: string }>(`/api/skills/${encodeURIComponent(name)}`),
  toggleSkill: (name: string, enabled: boolean) =>
    http(`/api/skills/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify({ enabled }),
    }),
}
