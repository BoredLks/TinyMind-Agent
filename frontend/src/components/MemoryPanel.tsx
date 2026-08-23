import { useEffect, useState } from 'react'
import { Plus, Trash2, X } from 'lucide-react'

import { api, type MemoryRow } from '../api/restClient'

export default function MemoryPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [rows, setRows] = useState<MemoryRow[]>([])
  const [narrative, setNarrative] = useState('')
  const [theme, setTheme] = useState('偏好')
  const [content, setContent] = useState('')
  const [notice, setNotice] = useState('')

  const load = async () => {
    try {
      const res = await api.listMemories()
      setRows(res.memories)
      setNarrative(res.narrative)
    } catch (err) {
      setNotice(err instanceof Error ? err.message : '加载记忆失败')
    }
  }

  const addMemory = async () => {
    const text = content.trim()
    if (!text) return
    try {
      await api.createMemory({ theme: theme.trim() || '偏好', content: text })
      setContent('')
      setNotice('记忆已新增')
      await load()
    } catch (err) {
      setNotice(err instanceof Error ? err.message : '新增记忆失败')
    }
  }

  useEffect(() => {
    if (open) void load()
  }, [open])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl bg-white p-5 text-gray-900 shadow-xl dark:bg-gray-800 dark:text-gray-100"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-base font-semibold">长期记忆</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200" title="关闭">
            <X size={18} />
          </button>
        </div>
        <div className="mb-4 rounded-lg border border-gray-200 bg-gray-50 p-3 dark:border-gray-700 dark:bg-gray-900">
          <div className="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">手动新增记忆</div>
          <div className="grid grid-cols-[120px_1fr_auto] gap-2">
            <input
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              className="rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-600 dark:bg-gray-800"
              placeholder="主题"
            />
            <input
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm dark:border-gray-600 dark:bg-gray-800"
              placeholder="例如：用户喜欢简洁回答"
              onKeyDown={(e) => {
                if (e.key === 'Enter') void addMemory()
              }}
            />
            <button
              onClick={() => void addMemory()}
              disabled={!content.trim()}
              className="inline-flex items-center gap-1 rounded-md bg-gray-900 px-3 text-sm text-white disabled:opacity-40 dark:bg-gray-100 dark:text-gray-900"
            >
              <Plus size={14} />
              新增
            </button>
          </div>
          {notice && <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">{notice}</div>}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {rows.length === 0 ? (
            <div className="rounded-lg border border-dashed border-gray-300 p-6 text-center text-sm text-gray-400 dark:border-gray-700">
              还没有长期记忆。
            </div>
          ) : (
            <div className="space-y-2">
              {rows.map((m) => (
                <div key={m.id} className="rounded-lg border border-gray-200 p-3 text-sm dark:border-gray-700">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-500 dark:bg-gray-900 dark:text-gray-300">
                      {m.theme}
                    </span>
                    <button
                      onClick={async () => {
                        try {
                          await api.deleteMemory(m.id)
                          setNotice('记忆已删除')
                          await load()
                        } catch (err) {
                          setNotice(err instanceof Error ? err.message : '删除记忆失败')
                        }
                      }}
                      className="text-gray-400 hover:text-red-500"
                      title="删除记忆"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <div>{m.content}</div>
                </div>
              ))}
            </div>
          )}
          {narrative && (
            <pre className="mt-4 whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-xs text-gray-600 dark:bg-gray-900 dark:text-gray-300">
              {narrative}
            </pre>
          )}
        </div>
        {rows.length > 0 && (
          <div className="mt-4 flex justify-end">
            <button
              onClick={async () => {
                try {
                  await api.clearMemories()
                  setNotice('记忆已清空')
                  await load()
                } catch (err) {
                  setNotice(err instanceof Error ? err.message : '清空失败')
                }
              }}
              className="rounded-lg border border-red-200 px-3 py-1.5 text-sm text-red-600 dark:border-red-900"
            >
              清空记忆
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
