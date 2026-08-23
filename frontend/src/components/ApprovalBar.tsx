import { Check, ShieldCheck, X } from 'lucide-react'

import { respondApproval } from '../api/wsClient'
import { useChatStore, type ToolItem } from '../stores/chatStore'

export default function ApprovalBar() {
  const approvalMode = useChatStore((s) => s.approvalMode)
  const items = useChatStore((s) => s.items)
  const pending = items.filter(
    (it): it is ToolItem => it.kind === 'tool' && it.state === 'awaiting_approval',
  )
  const pendingInteractions = items.filter(
    (it): it is ToolItem => it.kind === 'tool' && it.state === 'awaiting_user',
  )

  if (approvalMode !== 'require_approval') return null
  if (pending.length === 0 && pendingInteractions.length === 0) return null

  const approveAll = () => pending.forEach((p) => respondApproval(p.callId, true))
  const rejectAll = () => pending.forEach((p) => respondApproval(p.callId, false))

  return (
    <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 dark:border-amber-800 dark:bg-amber-950/30">
      {/* Tool approval requests */}
      {pending.length > 0 && (
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-amber-800 dark:text-amber-200">
            <ShieldCheck size={14} />
            <span>{pending.length} 个工具需要审批</span>
          </div>
          <button
            onClick={approveAll}
            className="inline-flex items-center gap-1 rounded-md bg-gray-900 px-2.5 py-1 text-xs text-white hover:bg-gray-700 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-300"
          >
            <Check size={12} />
            全部同意
          </button>
          <button
            onClick={rejectAll}
            className="inline-flex items-center gap-1 rounded-md border border-gray-300 px-2.5 py-1 text-xs hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
          >
            <X size={12} />
            全部拒绝
          </button>
          <div className="ml-auto flex flex-wrap gap-1.5">
            {pending.map((p) => (
              <span
                key={p.callId}
                className="inline-flex items-center gap-1 rounded-md bg-white px-2 py-0.5 text-[11px] text-gray-600 shadow-sm dark:bg-gray-800 dark:text-gray-300"
              >
                <span className="font-mono">{p.name}</span>
                <button
                  onClick={() => respondApproval(p.callId, true)}
                  className="rounded p-0.5 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30"
                  title="同意"
                >
                  <Check size={10} />
                </button>
                <button
                  onClick={() => respondApproval(p.callId, false)}
                  className="rounded p-0.5 text-red-600 hover:bg-red-100 dark:hover:bg-red-900/30"
                  title="拒绝"
                >
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Interaction requests (just show count) */}
      {pendingInteractions.length > 0 && (
        <div className="flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-300">
          <span>{pendingInteractions.length} 个工具正在等待你的输入（请在下方对话中回复）</span>
        </div>
      )}
    </div>
  )
}