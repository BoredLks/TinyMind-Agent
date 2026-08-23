import { useState } from 'react'
import { Layers, X, ChevronDown, ChevronUp } from 'lucide-react'

import ApprovalBar from './ApprovalBar'
import Composer from './Composer'
import MessageList from './MessageList'
import StatusIndicator from './StatusIndicator'
import { useChatStore } from '../stores/chatStore'
import type { ToolItem } from '../stores/chatStore'

export default function ChatView() {
  const [drawerOpen, setDrawerOpen] = useState(false)

  return (
    <main className="relative mx-auto flex w-full max-w-4xl flex-1 flex-col overflow-hidden">
      {/* Subagent button - top right */}
      <div className="absolute top-2 right-2 z-10">
        <SubagentButton open={drawerOpen} onToggle={() => setDrawerOpen(!drawerOpen)} />
      </div>

      <ApprovalBar />
      <MessageList />
      <StatusIndicator />
      <Composer />

      {/* Subagent Drawer */}
      {drawerOpen && <SubagentDrawer onClose={() => setDrawerOpen(false)} />}
    </main>
  )
}

function SubagentDisplay({ display }: { display: Record<string, unknown> }) {
  const d = display as Record<string, unknown>
  return (
    <div className="text-xs text-gray-500 space-y-0.5">
      {d.title != null && (
        <div className="flex items-center gap-1">
          <span className="text-gray-400">标题:</span>
          <span>{String(d.title ?? '')}</span>
        </div>
      )}
      {d.role != null && (
        <div className="flex items-center gap-1">
          <span className="text-gray-400">角色:</span>
          <span>{String(d.role ?? '')}</span>
        </div>
      )}
      {d.task != null && (
        <div className="mt-1">
          <span className="text-gray-400">任务:</span>
          <p className="text-gray-600 mt-0.5 line-clamp-3">{String(d.task ?? '')}</p>
        </div>
      )}
    </div>
  )
}

function SubagentButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const items = useChatStore((s) => s.items)
  const subagentCalls = items.filter(
    (it): it is ToolItem => it.kind === 'tool' && Boolean(it.subagent),
  )
  if (subagentCalls.length === 0 && !open) return null

  return (
    <button
      onClick={onToggle}
      className="flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white/90 px-2.5 py-1.5 text-xs text-gray-600 shadow-sm hover:bg-gray-50 backdrop-blur"
      title="查看子代理信息"
    >
      <Layers size={13} className="text-purple-500" />
      子代理 ({subagentCalls.length})
      {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
    </button>
  )
}

function SubagentDrawer({ onClose }: { onClose: () => void }) {
  const items = useChatStore((s) => s.items)
  const subagentCalls = items.filter(
    (it): it is ToolItem => it.kind === 'tool' && Boolean(it.subagent),
  )

  const STATE_LABEL: Record<string, string> = {
    running: '运行中',
    awaiting_approval: '待批准',
    awaiting_user: '等待输入',
    ok: '完成',
    error: '失败',
  }

  const STATE_BADGE: Record<string, string> = {
    running: 'bg-blue-100 text-blue-700',
    awaiting_approval: 'bg-amber-100 text-amber-700',
    awaiting_user: 'bg-amber-100 text-amber-700',
    ok: 'bg-green-100 text-green-700',
    error: 'bg-red-100 text-red-700',
  }

  return (
    <div className="absolute inset-0 z-20 flex flex-col bg-white/95 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 bg-gray-50/80">
        <div className="flex items-center gap-2">
          <Layers size={14} className="text-purple-500" />
          <span className="text-sm font-medium">子代理信息</span>
          <span className="text-[10px] bg-gray-200 text-gray-600 px-1.5 py-0.5 rounded">{subagentCalls.length}</span>
        </div>
        <button onClick={onClose} className="p-1 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-600">
          <X size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {subagentCalls.length === 0 ? (
          <div className="text-center text-sm text-gray-400 py-8">
            当前对话尚未调用子代理
          </div>
        ) : (
          subagentCalls.map((item) => (
            <div key={item.callId} className="rounded-lg border border-gray-200 bg-white p-3">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="font-mono text-xs font-medium text-gray-700">{item.name}</span>
                <span className={`rounded px-1.5 py-0.5 text-[10px] ${STATE_BADGE[item.state] || 'bg-gray-100 text-gray-500'}`}>
                  {STATE_LABEL[item.state] || item.state}
                </span>
              </div>
              {item.display && <SubagentDisplay display={item.display} />}
              {item.result && item.state !== 'running' && (
                <pre className="mt-2 max-h-24 overflow-auto rounded bg-gray-50 p-2 text-[11px] text-gray-600 border border-gray-100">
                  {item.result.length > 300 ? item.result.slice(0, 300) + '...' : item.result}
                </pre>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  )
}