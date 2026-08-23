import { useSessionsStore } from '../stores/sessionsStore'

const STAGES = [
  { key: 'brainstorm', label: '头脑风暴' },
  { key: 'plan', label: '写计划' },
  { key: 'execute', label: '执行' },
  { key: 'review', label: '评审' },
]

export default function StageBar() {
  const stage = useSessionsStore((s) => s.stage)
  if (!stage || stage === 'idle') return null

  const current = STAGES.findIndex((s) => s.key === stage)

  return (
    <div className="flex items-center gap-1 border-b border-gray-200 px-5 py-1.5 text-xs dark:border-gray-700">
      <span className="mr-1 text-gray-400">工作流</span>
      {STAGES.map((s, i) => (
        <span key={s.key} className="flex items-center gap-1">
          <span
            className={
              i === current
                ? 'rounded-full bg-gray-900 px-2 py-0.5 text-white dark:bg-gray-100 dark:text-gray-900'
                : i < current
                  ? 'text-gray-500'
                  : 'text-gray-300 dark:text-gray-600'
            }
          >
            {s.label}
          </span>
          {i < STAGES.length - 1 && <span className="text-gray-300 dark:text-gray-600">→</span>}
        </span>
      ))}
    </div>
  )
}
