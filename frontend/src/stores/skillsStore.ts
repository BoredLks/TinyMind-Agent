import { create } from 'zustand'

import { api, type SkillRow } from '../api/restClient'

interface SkillsState {
  skills: SkillRow[]
  bodies: Record<string, string>
  open: boolean
  setOpen: (open: boolean) => void
  load: () => Promise<void>
  toggle: (name: string, enabled: boolean) => Promise<void>
  loadBody: (name: string) => Promise<void>
}

export const useSkillsStore = create<SkillsState>((set, get) => ({
  skills: [],
  bodies: {},
  open: false,
  setOpen: (open) => set({ open }),
  load: async () => set({ skills: await api.listSkills() }),
  toggle: async (name, enabled) => {
    await api.toggleSkill(name, enabled)
    set((s) => ({ skills: s.skills.map((x) => (x.name === name ? { ...x, enabled } : x)) }))
  },
  loadBody: async (name) => {
    if (get().bodies[name]) return
    const detail = await api.getSkill(name)
    set((s) => ({ bodies: { ...s.bodies, [name]: detail.body } }))
  },
}))
