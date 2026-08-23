import { useChatStore } from '../stores/chatStore'
import { useSessionsStore } from '../stores/sessionsStore'
import { useSettingsStore } from '../stores/settingsStore'

// Dev: backend runs on 127.0.0.1:8000. Packaged app (M5): same-origin.
const WS_URL = import.meta.env.DEV
  ? 'ws://127.0.0.1:8000/ws/chat'
  : `ws://${location.host}/ws/chat`

let socket: WebSocket | null = null
let requestCounter = 0
// The in-flight request id per session ('' = ephemeral / no session). Events are
// applied only if they belong to the session's CURRENT request — this is what
// stops a cancelled/superseded stream from bleeding into a newer turn.
const activeRequests = new Map<string, string>()

interface ServerEvent {
  type:
    | 'status'
    | 'token'
    | 'reasoning'
    | 'tool_call'
    | 'tool_approval_request'
    | 'tool_result'
    | 'interaction_request'
    | 'sub_session_created'
    | 'sub_session_done'
    | 'workflow_stage'
    | 'retrying'
    | 'provider_fallback'
    | 'context_truncated'
    | 'cancelled'
    | 'done'
    | 'error'
  request_id?: string
  state?: string
  delta?: string
  message?: string
  call_id?: string
  name?: string
  args?: string
  ok?: boolean
  content?: string
  display?: Record<string, unknown>
  stage?: string
  attempt?: number
  provider_id?: string
  session_id?: string | null
  parent_session_id?: string
  title?: string
  role?: string
  task?: string
  depth?: number
  interaction_id?: string
  mode?: 'qa' | 'single_choice' | 'multi_choice' | 'confirm'
  question?: string
  options?: string[]
  code?: string
  usage?: { session_total: number; estimated: boolean }
  kept?: number
  total?: number
  subagent?: boolean
}

function viewedSid(): string {
  return useSessionsStore.getState().currentId ?? ''
}

function handleMessage(event: ServerEvent) {
  const store = useChatStore.getState()
  const sid = event.session_id ?? ''
  const isViewed = sid === viewedSid()
  // Does this event belong to the session's currently-tracked request?
  const isCurrentReq = activeRequests.get(sid) === event.request_id

  // The retry hint is a viewed-session concern; clear it on any other event
  // that belongs to the viewed session.
  if (event.type !== 'retrying' && isViewed) store.setRetrying(null)

  switch (event.type) {
    case 'retrying':
      if (isViewed) store.setRetrying(event.attempt ?? 1)
      break
    case 'status': {
      // A turn began for this session — record it and mark the session busy.
      if (event.request_id) activeRequests.set(sid, event.request_id)
      store.setGenerating(sid, true)
      if (isViewed) store.setNotice(null)
      break
    }
    case 'token': {
      if (!event.delta || !isCurrentReq) break
      if (isViewed) store.setReasoningActive(false)
      store.appendAssistantText(event.delta, sid)
      break
    }
    case 'reasoning': {
      if (!event.delta || !isCurrentReq) break
      if (isViewed) store.setReasoningActive(true)
      store.appendReasoning(event.delta, sid)
      break
    }
    case 'tool_call': {
      if (!isCurrentReq) break
      store.addToolCall(event.call_id ?? '', event.name ?? '', event.args ?? '', event.subagent ?? false, sid)
      break
    }
    case 'tool_approval_request': {
      if (store.approvalMode === 'auto_approve' && event.call_id) {
        respondApproval(event.call_id, true)
        break
      }
      if (!isCurrentReq) break
      store.setToolAwaitingApproval(event.call_id ?? '', sid)
      break
    }
    case 'tool_result': {
      if (!isCurrentReq) break
      store.setToolResult(event.call_id ?? '', Boolean(event.ok), event.content ?? '', event.display, sid)
      break
    }
    case 'interaction_request': {
      if (event.call_id && event.interaction_id && event.mode && event.question) {
        if (store.approvalMode === 'auto_approve' && event.mode === 'confirm') {
          respondInteraction(event.interaction_id, { approved: true, action: 'approve' })
          break
        }
        if (!isCurrentReq) break
        store.setToolInteraction(event.call_id, {
          interactionId: event.interaction_id,
          mode: event.mode,
          question: event.question,
          options: event.options ?? [],
          code: event.code,
        }, sid)
      }
      break
    }
    case 'sub_session_created':
      store.setNotice(`子代理会话已创建：${event.title ?? event.session_id ?? ''}`)
      void useSessionsStore.getState().refresh()
      break
    case 'sub_session_done':
      store.setNotice(event.ok === false ? '子代理执行失败' : '子代理执行完成')
      void useSessionsStore.getState().refresh()
      break
    case 'workflow_stage':
      if (event.stage && isViewed) useSessionsStore.getState().setStage(event.stage)
      break
    case 'context_truncated':
      if (isViewed) store.setNotice(`上下文较长，已保留最近 ${event.kept} 条（共 ${event.total} 条）`)
      break
    case 'provider_fallback':
      if (isViewed) store.setNotice(`当前模型连接失败，正在切换到备用模型 ${event.provider_id ?? ''}`)
      break
    case 'cancelled': {
      // Only the cancel of the session's CURRENT request resets its state; a
      // stale cancel from a superseded turn is ignored.
      if (!reqResolves(sid, event.request_id)) break
      activeRequests.delete(sid)
      store.setGenerating(sid, false)
      store.cancelPending(sid)
      if (isViewed) store.setReasoningActive(false)
      break
    }
    case 'done': {
      if (!reqResolves(sid, event.request_id)) {
        void useSessionsStore.getState().refresh()
        break
      }
      activeRequests.delete(sid)
      store.setGenerating(sid, false)
      if (isViewed) {
        store.setReasoningActive(false)
        if (event.usage) store.setUsage(event.usage)
      }
      void useSessionsStore.getState().refresh()
      break
    }
    case 'error': {
      if (!reqResolves(sid, event.request_id)) break
      activeRequests.delete(sid)
      store.setGenerating(sid, false)
      if (isViewed) {
        store.setError(event.message ?? 'Unknown error')
        store.setReasoningActive(false)
      } else {
        store.setNotice(`会话生成失败：${event.message ?? ''}`)
      }
      break
    }
  }
}

/**
 * A terminal event (done/cancelled/error) should only mutate a session's state
 * if it matches the session's current request, OR the session has no tracked
 * request anymore. This prevents a superseded turn from clobbering the state of
 * the newer turn that replaced it.
 */
function reqResolves(sid: string, requestId: string | undefined): boolean {
  const current = activeRequests.get(sid)
  return current === undefined || current === requestId
}

function ensureSocket(): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      resolve(socket)
      return
    }
    const ws = new WebSocket(WS_URL)
    socket = ws
    ws.onopen = () => {
      useChatStore.getState().setWsConnected(true)
      resolve(ws)
    }
    ws.onerror = () => {
      useChatStore.getState().setWsConnected(false)
      reject(new Error('无法连接到后端 WebSocket'))
    }
    ws.onmessage = (ev) => handleMessage(JSON.parse(ev.data) as ServerEvent)
    ws.onclose = () => {
      socket = null
      useChatStore.getState().setWsConnected(false)
      // The backend cancels all in-flight turns on disconnect; clear local
      // generating state so the UI doesn't get stuck showing "生成中".
      const store = useChatStore.getState()
      for (const sid of activeRequests.keys()) store.setGenerating(sid, false)
      activeRequests.clear()
    }
  })
}

export async function sendMessage(content: string, providerId?: string): Promise<void> {
  const store = useChatStore.getState()
  const currentId = useSessionsStore.getState().currentId
  const sid = currentId ?? ''
  const requestId = `r${++requestCounter}`
  // Record the request BEFORE sending so the first events route correctly.
  activeRequests.set(sid, requestId)
  store.pushUser(content, sid)
  store.setGenerating(sid, true)
  const activeProviderId = providerId || useSettingsStore.getState().settings?.active_provider_id
  try {
    const ws = await ensureSocket()
    ws.send(
      JSON.stringify({
        type: 'user_message',
        request_id: requestId,
        session_id: currentId,
        provider_id: activeProviderId,
        content,
      }),
    )
  } catch (err) {
    activeRequests.delete(sid)
    store.setGenerating(sid, false)
    store.setError(err instanceof Error ? err.message : '连接失败')
  }
}

export function respondApproval(callId: string, approved: boolean): void {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'tool_approval_response', call_id: callId, approved }))
  }
}

export function respondInteraction(interactionId: string, response: unknown): void {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'interaction_response', interaction_id: interactionId, response }))
  }
}

/** Cancel the in-flight generation of the currently-viewed session. */
export function cancelGeneration(): void {
  const sid = viewedSid()
  const rid = activeRequests.get(sid)
  if (socket && socket.readyState === WebSocket.OPEN && rid) {
    socket.send(JSON.stringify({ type: 'cancel', request_id: rid }))
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  const baseUrl = import.meta.env.DEV ? 'http://127.0.0.1:8000' : ''
  try {
    const res = await fetch(`${baseUrl}/api/health`, { signal: AbortSignal.timeout(3000) })
    const ok = res.ok
    useChatStore.getState().setWsConnected(ok)
    return ok
  } catch {
    useChatStore.getState().setWsConnected(false)
    return false
  }
}
