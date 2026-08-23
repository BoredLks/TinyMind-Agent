import { useEffect, useState } from 'react'

import { useToolsStore } from '../stores/toolsStore'

export default function ToolsPanel() {
  const open = useToolsStore((s) => s.open)
  const setOpen = useToolsStore((s) => s.setOpen)
  const tools = useToolsStore((s) => s.tools)
  const loadedPlugins = useToolsStore((s) => s.loadedPlugins)
  const docs = useToolsStore((s) => s.docs)
  const load = useToolsStore((s) => s.load)
  const loadDoc = useToolsStore((s) => s.loadDoc)
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    if (open) void load()
  }, [open, load])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
      onClick={() => setOpen(false)}
    >
      <div
        className="max-h-[85vh] w-[560px] overflow-y-auto rounded-2xl bg-white p-6 text-gray-900 shadow-xl dark:bg-gray-800 dark:text-gray-100"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="mb-1 text-lg font-semibold">工具</h2>
        <p className="mb-4 text-xs text-gray-400">
          当前可用的工具列表。有副作用的工具（写文件、执行命令等）需要用户审批后才会执行。
        </p>

        {loadedPlugins.length > 0 && (
          <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-700 dark:border-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
            已加载 {loadedPlugins.length} 个外部插件
          </div>
        )}

        <div className="space-y-2">
          {tools.map((tool) => (
            <div key={tool.name} className="rounded-xl border border-gray-200 p-3 dark:border-gray-700">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm">{tool.name}</span>
                    {tool.requires_approval && (
                      <span className="rounded bg-amber-100 px-1.5 text-[11px] text-amber-700 dark:bg-amber-900/40 dark:text-amber-400">
                        需审批
                      </span>
                    )}
                    {tool.has_doc && (
                      <span className="rounded bg-gray-100 px-1.5 text-[11px] text-gray-500 dark:bg-gray-700">
                        有文档
                      </span>
                    )}
                    {tool.external && (
                      <span className="rounded bg-green-100 px-1.5 text-[11px] text-green-700 dark:bg-green-900/40 dark:text-green-400">
                        外部加载
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{tool.description}</p>
                </div>
              </div>
              {tool.has_doc && (
                <button
                  aria-label={`${expanded === tool.name ? '收起' : '查看'} ${tool.name} 文档`}
                  className="mt-2 text-xs text-gray-400 hover:text-gray-600"
                  onClick={async () => {
                    if (expanded === tool.name) {
                      setExpanded(null)
                    } else {
                      await loadDoc(tool.name)
                      setExpanded(tool.name)
                    }
                  }}
                >
                  {expanded === tool.name ? '收起文档' : '查看文档'}
                </button>
              )}
              {expanded === tool.name && docs[tool.name] && (
                <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded bg-gray-50 p-2 text-xs dark:bg-gray-900">
                  {docs[tool.name]}
                </pre>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={() => setOpen(false)}
            className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm dark:border-gray-600"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  )
}