import { create } from 'zustand'

export interface PreviewFile {
  path: string
  content: string
  kind: 'code' | 'html' | 'text'
}

interface PreviewState {
  file: PreviewFile | null
  open: (file: PreviewFile) => void
  close: () => void
}

export const usePreviewStore = create<PreviewState>((set) => ({
  file: null,
  open: (file) => set({ file }),
  close: () => set({ file: null }),
}))