import { create } from 'zustand'

import { api } from '../api/restClient'

export interface ToolRow {
  name: string
  description: string
  requires_approval: boolean
  has_doc: boolean
  external?: boolean
}

interface ToolsState {
  tools: ToolRow[]
  loadedPlugins: string[]
  docs: Record<string, string>
  open: boolean
  setOpen: (open: boolean) => void
  load: () => Promise<void>
  loadDoc: (name: string) => Promise<void>
}

export const useToolsStore = create<ToolsState>((set, get) => ({
  tools: [],
  loadedPlugins: [],
  docs: {},
  open: false,
  setOpen: (open) => set({ open }),
  load: async () => {
    const res = await api.listTools()
    set({ tools: res.tools, loadedPlugins: res.loaded_plugins })
  },
  loadDoc: async (name: string) => {
    if (!name || get().docs[name]) return
    const res = await api.getToolDoc(name)
    set((s) => ({ docs: { ...s.docs, [name]: res.doc } }))
  },
}))
