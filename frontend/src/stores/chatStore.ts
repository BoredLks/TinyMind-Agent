import { create } from 'zustand'

export type Role = 'user' | 'assistant'
export type Status = 'idle' | 'generating' | 'error'
export type ApprovalMode = 'auto_approve' | 'require_approval'
export type ToolState = 'running' | 'awaiting_approval' | 'awaiting_user' | 'ok' | 'error'

export interface MessageItem {
  kind: 'message'
  id: string
  role: Role
  content: string
  reasoning?: string
}

export interface ToolItem {
  kind: 'tool'
  id: string
  callId: string
  name: string
  args: string
  state: ToolState
  result?: string
  display?: Record<string, unknown>
  interaction?: {
    interactionId: string
    mode: 'qa' | 'single_choice' | 'multi_choice' | 'confirm'
    question: string
    options: string[]
    code?: string
  }
  subagent?: boolean
}

export type TimelineItem = MessageItem | ToolItem

export interface Usage {
  session_total: number
  estimated: boolean
}

interface ChatState {
  items: TimelineItem[]
  status: Status
  approvalMode: ApprovalMode
  error: string | null
  retrying: number | null
  usage: Usage | null
  notice: string | null
  wsConnected: boolean
  reasoningActive: boolean
  // The session whose timeline is mirrored in `items`. '' means ephemeral/none.
  viewedId: string
  // Which sessions currently have a running turn (keyed by session id, '' = ephemeral).
  generating: Record<string, boolean>
  // Live timelines for sessions that are generating but NOT currently viewed.
  // Keyed by session id; merged back into `items` when the session is viewed.
  _liveItems: Record<string, TimelineItem[]>
  setWsConnected: (connected: boolean) => void
  setReasoningActive: (active: boolean) => void
  setViewedId: (sessionId: string) => void
  setGenerating: (sessionId: string, value: boolean) => void
  isGenerating: (sessionId: string) => boolean
  syncViewedStatus: () => void
  saveLiveItems: (sessionId: string) => void
  restoreLiveItems: (sessionId: string) => boolean
  clearLiveItems: (sessionId: string) => void
  setHistory: (rows: { id: string; role: string; content: string }[]) => void
  pushUser: (content: string, sessionId?: string) => void
  appendAssistantText: (delta: string, sessionId?: string) => void
  appendReasoning: (delta: string, sessionId?: string) => void
  addToolCall: (callId: string, name: string, args: string, subagent?: boolean, sessionId?: string) => void
  setToolAwaitingApproval: (callId: string, sessionId?: string) => void
  setToolInteraction: (callId: string, interaction: NonNullable<ToolItem['interaction']>, sessionId?: string) => void
  setToolResult: (callId: string, ok: boolean, result: string, display?: Record<string, unknown>, sessionId?: string) => void
  cancelPending: (sessionId?: string) => void
  setStatus: (status: Status) => void
  setError: (error: string) => void
  setRetrying: (attempt: number | null) => void
  setUsage: (usage: Usage) => void
  setNotice: (notice: string | null) => void
  setApprovalMode: (mode: ApprovalMode) => void
  reset: () => void
}

// ---- pure timeline transforms (operate on a copy, return a new array) ----

function appendText(arr: TimelineItem[], delta: string): TimelineItem[] {
  const items = arr.slice()
  const last = items[items.length - 1]
  if (last && last.kind === 'message' && last.role === 'assistant') {
    items[items.length - 1] = { ...last, content: last.content + delta }
  } else {
    items.push({ kind: 'message', id: crypto.randomUUID(), role: 'assistant', content: delta })
  }
  return items
}

function appendReasoningText(arr: TimelineItem[], delta: string): TimelineItem[] {
  const items = arr.slice()
  const last = items[items.length - 1]
  if (last && last.kind === 'message' && last.role === 'assistant') {
    items[items.length - 1] = { ...last, reasoning: (last.reasoning || '') + delta }
  } else {
    items.push({ kind: 'message', id: crypto.randomUUID(), role: 'assistant', content: '', reasoning: delta })
  }
  return items
}

export const useChatStore = create<ChatState>((set, get) => {
  /**
   * Route a timeline mutation to the right place: the live `items` array when
   * the target session is the one being viewed, otherwise that session's
   * background buffer in `_liveItems`. This is what keeps two concurrent
   * conversations from bleeding into each other.
   */
  const mutate = (
    sessionId: string | undefined,
    fn: (arr: TimelineItem[]) => TimelineItem[],
  ) =>
    set((s) => {
      const sid = sessionId ?? s.viewedId
      if (sid === s.viewedId) {
        return { items: fn(s.items) }
      }
      const buf = s._liveItems[sid] ?? []
      return { _liveItems: { ...s._liveItems, [sid]: fn(buf) } }
    })

  return {
    items: [],
    status: 'idle',
    approvalMode: 'require_approval',
    error: null,
    retrying: null,
    usage: null,
    notice: null,
    wsConnected: false,
    reasoningActive: false,
    viewedId: '',
    generating: {},
    _liveItems: {},

    setViewedId: (sessionId) => set({ viewedId: sessionId }),

    setGenerating: (sessionId, value) =>
      set((s) => {
        const generating = { ...s.generating }
        if (value) generating[sessionId] = true
        else delete generating[sessionId]
        const next: Partial<ChatState> = { generating }
        // Keep the viewed status in sync with the viewed session's state.
        if (sessionId === s.viewedId) {
          if (value) next.status = 'generating'
          else if (s.status !== 'error') next.status = 'idle'
        }
        return next
      }),

    isGenerating: (sessionId) => Boolean(get().generating[sessionId]),

    syncViewedStatus: () =>
      set((s) => ({
        status: s.generating[s.viewedId] ? 'generating' : 'idle',
        error: null,
      })),

    saveLiveItems: (sessionId) =>
      set((s) => ({
        _liveItems: { ...s._liveItems, [sessionId]: [...s.items] },
      })),

    restoreLiveItems: (sessionId) => {
      const state = get()
      const cached = state._liveItems[sessionId]
      if (!cached) return false
      const { [sessionId]: _removed, ...rest } = state._liveItems
      set({ items: cached, _liveItems: rest })
      return true
    },

    /** Drop a session's cached live timeline (e.g. on delete). */
    clearLiveItems: (sessionId: string) =>
      set((s) => {
        const { [sessionId]: _removed, ...rest } = s._liveItems
        return { _liveItems: rest }
      }),

    setHistory: (rows) => {
      const items: TimelineItem[] = []
      for (const r of rows) {
        if (r.role === 'tool_event') {
          try {
            const ev = JSON.parse(r.content) as { type: string; call_id?: string; name?: string; args?: string; ok?: boolean; content?: string; display?: Record<string, unknown> }
            if (ev.type === 'tool_call' && ev.call_id) {
              items.push({
                kind: 'tool',
                id: r.id,
                callId: ev.call_id,
                name: ev.name ?? '',
                args: ev.args ?? '',
                state: 'ok',
                subagent: false,
              })
            } else if (ev.type === 'tool_result' && ev.call_id) {
              const existing = items.find((it) => it.kind === 'tool' && it.callId === ev.call_id) as ToolItem | undefined
              if (existing) {
                existing.state = ev.ok ? 'ok' : 'error'
                existing.result = ev.content
                existing.display = ev.display
              }
            }
          } catch {
            // skip malformed tool events
          }
        } else {
          items.push({
            kind: 'message',
            id: r.id,
            role: r.role as Role,
            content: r.content,
          })
        }
      }
      set({
        items,
        error: null,
        retrying: null,
        usage: null,
        notice: null,
        reasoningActive: false,
      })
    },

    pushUser: (content, sessionId) =>
      mutate(sessionId, (arr) => [
        ...arr,
        { kind: 'message', id: crypto.randomUUID(), role: 'user', content },
      ]),

    appendAssistantText: (delta, sessionId) => mutate(sessionId, (arr) => appendText(arr, delta)),

    appendReasoning: (delta, sessionId) => mutate(sessionId, (arr) => appendReasoningText(arr, delta)),

    addToolCall: (callId, name, args, subagent = false, sessionId) =>
      mutate(sessionId, (arr) => [
        ...arr,
        { kind: 'tool', id: crypto.randomUUID(), callId, name, args, state: 'running', subagent },
      ]),

    setToolAwaitingApproval: (callId, sessionId) =>
      mutate(sessionId, (arr) =>
        arr.map((it) =>
          it.kind === 'tool' && it.callId === callId ? { ...it, state: 'awaiting_approval' } : it,
        ),
      ),

    setToolInteraction: (callId, interaction, sessionId) =>
      mutate(sessionId, (arr) =>
        arr.map((it) =>
          it.kind === 'tool' && it.callId === callId
            ? { ...it, state: 'awaiting_user', interaction }
            : it,
        ),
      ),

    setToolResult: (callId, ok, result, display, sessionId) =>
      mutate(sessionId, (arr) =>
        arr.map((it) =>
          it.kind === 'tool' && it.callId === callId
            ? { ...it, state: ok ? 'ok' : 'error', result, display, interaction: undefined }
            : it,
        ),
      ),

    // Finalize any still-pending tool cards (running / awaiting) when a turn is
    // cancelled, so they don't linger forever in a "运行中" / "待批准" state.
    cancelPending: (sessionId) =>
      mutate(sessionId, (arr) =>
        arr.map((it) =>
          it.kind === 'tool' && (it.state === 'running' || it.state === 'awaiting_approval' || it.state === 'awaiting_user')
            ? { ...it, state: 'error', result: it.result ?? '已取消', interaction: undefined }
            : it,
        ),
      ),

    setStatus: (status) => set({ status }),
    setError: (error) => set({ error, status: 'error', retrying: null }),
    setRetrying: (attempt) => set({ retrying: attempt }),
    setUsage: (usage) => set({ usage }),
    setNotice: (notice) => set({ notice }),
    setWsConnected: (connected) => set({ wsConnected: connected }),
    setReasoningActive: (active) => set({ reasoningActive: active }),
    setApprovalMode: (mode) => set({ approvalMode: mode }),
    reset: () =>
      // Note: approvalMode is intentionally preserved across resets/new sessions
      // so a user's "全部同意" choice isn't silently reverted on every new chat.
      set({ items: [], status: 'idle', error: null, retrying: null, usage: null, notice: null, reasoningActive: false }),
  }
})
