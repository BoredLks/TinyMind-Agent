import { useEffect, useState } from 'react'

import { useSkillsStore } from '../stores/skillsStore'

export default function SkillsPanel() {
  const open = useSkillsStore((s) => s.open)
  const setOpen = useSkillsStore((s) => s.setOpen)
  const skills = useSkillsStore((s) => s.skills)
  const bodies = useSkillsStore((s) => s.bodies)
  const load = useSkillsStore((s) => s.load)
  const toggle = useSkillsStore((s) => s.toggle)
  const loadBody = useSkillsStore((s) => s.loadBody)
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
        <h2 className="mb-1 text-lg font-semibold">技能</h2>
        <p className="mb-4 text-xs text-gray-400">
          启用的技能会进入助手的系统提示；需要时助手用 load_skill 加载全文并遵循。
        </p>

        <div className="space-y-2">
          {skills.map((sk) => (
            <div key={sk.name} className="rounded-xl border border-gray-200 p-3 dark:border-gray-700">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm">{sk.name}</span>
                    <span className="rounded bg-gray-100 px-1.5 text-[11px] text-gray-500 dark:bg-gray-700">
                      {sk.source}
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{sk.description}</p>
                </div>
                <label className="flex shrink-0 cursor-pointer items-center gap-1 text-xs">
                  <input
                    type="checkbox"
                    aria-label={`启用 ${sk.name}`}
                    checked={sk.enabled}
                    onChange={(e) => toggle(sk.name, e.target.checked)}
                  />
                  启用
                </label>
              </div>
              <button
                aria-label={`${expanded === sk.name ? '收起' : '查看'} ${sk.name} 内容`}
                className="mt-2 text-xs text-gray-400 hover:text-gray-600"
                onClick={async () => {
                  if (expanded === sk.name) {
                    setExpanded(null)
                  } else {
                    await loadBody(sk.name)
                    setExpanded(sk.name)
                  }
                }}
              >
                {expanded === sk.name ? '收起' : '查看内容'}
              </button>
              {expanded === sk.name && bodies[sk.name] && (
                <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded bg-gray-50 p-2 text-xs dark:bg-gray-900">
                  {bodies[sk.name]}
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
