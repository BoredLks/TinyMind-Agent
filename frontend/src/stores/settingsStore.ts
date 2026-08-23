import { create } from 'zustand'

import { api } from '../api/restClient'
import type { ProviderRow } from '../api/restClient'

export interface AppSettings {
  base_url: string
  model: string
  active_provider_id: string
  providers: ProviderRow[]
  temperature: number
  top_p: number | null
  max_tokens: number | null
  system_prompt: string | null
  context_max_messages: number
  context_strategy: string
  context_max_tokens: number
  max_tool_iterations: number
  max_tool_arg_len: number
  max_tool_result_len: number
  max_tool_text_len: number
  api_timeout: number
  turn_timeout: number
  theme: 'light' | 'dark'
  workspace_root: string
  current_project_id: string | null
  tools_enabled: boolean
  disabled_tools: string[]
  memory_enabled: boolean
  has_api_key: boolean
}

interface SettingsState {
  settings: AppSettings | null
  open: boolean
  setOpen: (open: boolean) => void
  load: () => Promise<void>
  save: (patch: Partial<AppSettings>) => Promise<void>
  setApiKey: (key: string) => Promise<void>
  clearApiKey: () => Promise<void>
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  settings: null,
  open: false,
  setOpen: (open) => set({ open }),
  load: async () => set({ settings: (await api.getSettings()) as unknown as AppSettings }),
  save: async (patch) =>
    set({ settings: (await api.updateSettings(patch)) as unknown as AppSettings }),
  setApiKey: async (key) => {
    await api.setApiKey(key)
    await get().load()
  },
  clearApiKey: async () => {
    await api.clearApiKey()
    await get().load()
  },
}))
