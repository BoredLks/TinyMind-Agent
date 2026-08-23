import { create } from 'zustand'

import { api, type SessionMeta } from '../api/restClient'
import { useChatStore } from './chatStore'

interface SessionsState {
  sessions: SessionMeta[]
  currentId: string | null
  stage: string
  bootstrap: () => Promise<void>
  select: (id: string) => Promise<void>
  create: (projectId?: string | null) => Promise<void>
  refresh: () => Promise<void>
  rename: (id: string, title: string) => Promise<void>
  remove: (id: string) => Promise<void>
  undoLastTurn: () => Promise<void>
  setStage: (stage: string) => void
}

// Guards against React StrictMode double-invoking bootstrap in dev (which would
// otherwise create duplicate empty sessions).
let bootstrapped = false

async function loadHistory(id: string): Promise<void> {
  const rows = await api.getMessages(id)
  useChatStore.getState().setHistory(rows)
}

export const useSessionsStore = create<SessionsState>((set, get) => ({
  sessions: [],
  currentId: null,
  stage: 'idle',

  bootstrap: async () => {
    if (bootstrapped || get().currentId) return
    bootstrapped = true
    let list = await api.listSessions()
    if (list.length === 0) {
      list = [await api.createSession()]
    }
    set({ sessions: list, currentId: list[0].id, stage: list[0].workflow_stage ?? 'idle' })
    useChatStore.getState().setViewedId(list[0].id)
    await loadHistory(list[0].id)
  },

  select: async (id) => {
    if (get().currentId === id) return
    const chatStore = useChatStore.getState()
    const prevId = get().currentId
    // Snapshot the outgoing session's live timeline if it is still generating,
    // so its background stream keeps accumulating into its own buffer.
    if (prevId && chatStore.isGenerating(prevId)) {
      chatStore.saveLiveItems(prevId)
    }
    const found = get().sessions.find((s) => s.id === id)
    set({ currentId: id, stage: found?.workflow_stage ?? 'idle' })
    chatStore.setViewedId(id)
    // Restore the target session's buffered live timeline, else load from DB.
    if (!chatStore.restoreLiveItems(id)) {
      await loadHistory(id)
    }
    // Reflect the target session's own generating state in the viewed status.
    chatStore.syncViewedStatus()
    chatStore.setReasoningActive(false)
  },

  create: async (projectId?: string | null) => {
    const prevId = get().currentId
    const chatStore = useChatStore.getState()
    // Save live items if the previous session is still generating.
    if (prevId && chatStore.isGenerating(prevId)) {
      chatStore.saveLiveItems(prevId)
    }
    const session = await api.createSession('新会话', projectId ?? null)
    set((state) => ({
      sessions: [session, ...state.sessions],
      currentId: session.id,
      stage: 'idle',
    }))
    chatStore.reset()
    chatStore.setViewedId(session.id)
    chatStore.syncViewedStatus()
  },

  refresh: async () => {
    const list = await api.listSessions()
    const currentId = get().currentId ?? list[0]?.id ?? null
    const current = list.find((s) => s.id === currentId)
    set({ sessions: list, currentId, stage: current?.workflow_stage ?? get().stage })
  },

  rename: async (id, title) => {
    const session = await api.renameSession(id, title)
    set((state) => ({
      sessions: state.sessions.map((s) => (s.id === id ? { ...s, title: session.title } : s)),
    }))
  },

  remove: async (id) => {
    await api.deleteSession(id)
    const chatStore = useChatStore.getState()
    chatStore.setGenerating(id, false)
    chatStore.clearLiveItems(id)
    const remaining = get().sessions.filter((s) => s.id !== id)
    set({ sessions: remaining })
    if (get().currentId === id) {
      if (remaining.length > 0) {
        await get().select(remaining[0].id)
      } else {
        await get().create()
      }
    }
  },

  undoLastTurn: async () => {
    const id = get().currentId
    if (!id) return
    const res = await api.undoSession(id, 1)
    const chatStore = useChatStore.getState()
    chatStore.setHistory(res.messages)
    chatStore.syncViewedStatus()
  },

  setStage: (stage) => set({ stage }),
}))
