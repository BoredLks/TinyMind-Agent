import { create } from 'zustand'

import { api, type ProjectRow } from '../api/restClient'
import { useSettingsStore } from './settingsStore'

interface ProjectsState {
  projects: ProjectRow[]
  currentId: string | null
  error: string | null
  load: () => Promise<void>
  add: (path: string, name?: string) => Promise<void>
  select: (id: string) => Promise<void>
  rename: (id: string, name: string) => Promise<void>
  remove: (id: string) => Promise<void>
  clearError: () => void
}

function markCurrent(projects: ProjectRow[], currentId: string | null): ProjectRow[] {
  return projects.map((p) => ({ ...p, current: p.id === currentId }))
}

export const useProjectsStore = create<ProjectsState>((set, get) => ({
  projects: [],
  currentId: null,
  error: null,

  load: async () => {
    const res = await api.listProjects()
    set({
      projects: markCurrent(res.projects, res.current_project_id),
      currentId: res.current_project_id,
      error: null,
    })
  },

  add: async (path, name) => {
    try {
      const project = await api.createProject(path, name)
      await useSettingsStore.getState().load()
      set((state) => ({
        projects: markCurrent(
          [project, ...state.projects.filter((p) => p.id !== project.id)],
          project.id,
        ),
        currentId: project.id,
        error: null,
      }))
    } catch (exc) {
      set({ error: exc instanceof Error ? exc.message : String(exc) })
    }
  },

  select: async (id) => {
    try {
      const project = await api.selectProject(id)
      await useSettingsStore.getState().load()
      set((state) => ({
        projects: markCurrent(
          [project, ...state.projects.filter((p) => p.id !== project.id)],
          project.id,
        ),
        currentId: project.id,
        error: null,
      }))
    } catch (exc) {
      set({ error: exc instanceof Error ? exc.message : String(exc) })
    }
  },

  rename: async (id, name) => {
    const project = await api.renameProject(id, name)
    set((state) => ({
      projects: state.projects.map((p) => (p.id === id ? { ...p, name: project.name } : p)),
    }))
  },

  remove: async (id) => {
    await api.deleteProject(id)
    await get().load()
    await useSettingsStore.getState().load()
  },

  clearError: () => set({ error: null }),
}))
