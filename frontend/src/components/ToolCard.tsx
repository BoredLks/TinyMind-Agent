import { useState } from 'react'
import { BookOpen, Check, ChevronDown, ChevronUp, Eye, FileText, Play, Terminal, Wrench, X } from 'lucide-react'

import { respondApproval, respondInteraction } from '../api/wsClient'
import { usePreviewStore } from '../stores/previewStore'
import type { ToolItem } from '../stores/chatStore'

const LABEL: Record<ToolItem['state'], string> = {
  running: '运行中',
  awaiting_approval: '待批准',
  awaiting_user: '等待输入',
  ok: '成功',
  error: '失败',
}

const BADGE: Record<ToolItem['state'], string> = {
  running: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200',
  awaiting_approval: 'bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-200',
  awaiting_user: 'bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-200',
  ok: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-200',
  error: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-200',
}

function guessKind(path: string): 'code' | 'html' | 'text' {
  const lower = path.toLowerCase()
  if (lower.endsWith('.html') || lower.endsWith('.htm')) return 'html'
  const codeExts = ['.js', '.ts', '.tsx', '.jsx', '.py', '.java', '.c', '.cpp', '.h', '.css', '.json', '.yaml', '.yml', '.toml', '.xml', '.sh', '.bat', '.ps1', '.sql', '.md']
  if (codeExts.some((ext) => lower.endsWith(ext))) return 'code'
  return 'text'
}

export default function ToolCard({ item }: { item: ToolItem }) {
  const [open, setOpen] = useState(item.state === 'awaiting_approval' && !item.subagent)
  const [answer, setAnswer] = useState('')
  const [single, setSingle] = useState(item.interaction?.options?.[0] ?? '')
  const [multi, setMulti] = useState<string[]>([])
  const [submitted, setSubmitted] = useState(false)
  const previewOpen = usePreviewStore((s) => s.open)
  const ToolIcon = item.name === 'load_skill' ? BookOpen : Wrench

  return (
    <div className="my-2 rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 font-mono text-xs text-gray-600 dark:text-gray-300">
            <ToolIcon size={14} strokeWidth={1.8} />
            {item.name}
          </span>
          <span className={`rounded px-1.5 py-0.5 text-[11px] ${BADGE[item.state]}`}>
            {LABEL[item.state]}
          </span>
          {item.subagent && (
            <span className="rounded bg-purple-100 px-1.5 py-0.5 text-[11px] text-purple-700 dark:bg-purple-900 dark:text-purple-200">
              子代理
            </span>
          )}
        </div>
        <button
          onClick={() => setOpen((o) => !o)}
          className="inline-flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
          title={open ? '收起详情' : '展开详情'}
        >
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {open ? '收起' : '详情'}
        </button>
      </div>

      {open && (
        <pre className="mt-2 overflow-x-auto rounded-md bg-white p-2 text-xs text-gray-700 dark:bg-gray-900 dark:text-gray-300">
          {item.args}
        </pre>
      )}

      {item.state === 'awaiting_approval' && !item.subagent && (
        <div className="mt-2 flex gap-2">
          <button
            onClick={() => respondApproval(item.callId, true)}
            className="inline-flex items-center gap-1 rounded-md bg-gray-900 px-3 py-1 text-xs text-white dark:bg-gray-100 dark:text-gray-900"
          >
            <Check size={13} />
            允许
          </button>
          <button
            onClick={() => respondApproval(item.callId, false)}
            className="inline-flex items-center gap-1 rounded-md border border-gray-300 px-3 py-1 text-xs dark:border-gray-600"
          >
            <X size={13} />
            拒绝
          </button>
        </div>
      )}

      {item.state === 'awaiting_user' && item.interaction && (
        <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs dark:border-amber-900 dark:bg-amber-950/30">
          <div className="mb-2 font-medium text-amber-800 dark:text-amber-100">
            {item.interaction.question}
          </div>
          {item.interaction.code && (
            <pre className="mb-3 max-h-60 overflow-auto rounded-md bg-gray-950 p-3 text-[11px] text-gray-100">
              {item.interaction.code}
            </pre>
          )}
          {item.interaction.mode === 'qa' && (
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              rows={3}
              className="w-full resize-none rounded-md border border-gray-300 bg-white px-2 py-1.5 text-xs dark:border-gray-600 dark:bg-gray-800"
              placeholder="输入回复"
            />
          )}
          {item.interaction.mode === 'single_choice' && (
            <div className="space-y-1.5">
              {item.interaction.options.map((opt) => (
                <label key={opt} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="radio"
                    checked={(single || item.interaction?.options?.[0]) === opt}
                    onChange={() => setSingle(opt)}
                  />
                  <span>{opt}</span>
                </label>
              ))}
            </div>
          )}
          {item.interaction.mode === 'multi_choice' && (
            <div className="space-y-1.5">
              {item.interaction.options.map((opt) => (
                <label key={opt} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="checkbox"
                    checked={multi.includes(opt)}
                    onChange={(e) =>
                      setMulti((cur) => (e.target.checked ? [...cur, opt] : cur.filter((x) => x !== opt)))
                    }
                  />
                  <span>{opt}</span>
                </label>
              ))}
            </div>
          )}
          <div className="mt-3 flex gap-2">
            {item.interaction.mode === 'confirm' ? (
              <>
                <button
                  disabled={submitted}
                  onClick={() => {
                    setSubmitted(true)
                    respondInteraction(item.interaction!.interactionId, { approved: true, action: 'approve' })
                  }}
                  className="inline-flex items-center gap-1 rounded-md bg-gray-900 px-3 py-1 text-xs text-white disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900"
                >
                  <Play size={13} />
                  执行
                </button>
                <button
                  disabled={submitted}
                  onClick={() => {
                    setSubmitted(true)
                    respondInteraction(item.interaction!.interactionId, { approved: false, action: 'reject' })
                  }}
                  className="inline-flex items-center gap-1 rounded-md border border-gray-300 px-3 py-1 text-xs disabled:opacity-50 dark:border-gray-600"
                >
                  <X size={13} />
                  取消
                </button>
              </>
            ) : (
              <button
                disabled={submitted || (item.interaction.mode === 'qa' && !answer.trim())}
                onClick={() => {
                  setSubmitted(true)
                  const response =
                    item.interaction?.mode === 'qa'
                      ? answer.trim()
                      : item.interaction?.mode === 'single_choice'
                        ? single || item.interaction.options[0]
                        : multi
                  respondInteraction(item.interaction!.interactionId, response)
                }}
                className="inline-flex items-center gap-1 rounded-md bg-gray-900 px-3 py-1 text-xs text-white disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900"
              >
                <Check size={13} />
                提交
              </button>
            )}
          </div>
        </div>
      )}

      {item.display && item.state !== 'awaiting_user' && (
        <StructuredDisplay
          display={item.display}
          onPreview={item.display?.kind === 'file' ? (content, path) => {
            const previewContent = content || String(item.display?.previewContent || item.result || '')
            if (previewContent) {
              previewOpen({ path, content: previewContent, kind: guessKind(path) })
            }
          } : undefined}
        />
      )}

      {open && item.result && (
        <pre className="mt-2 overflow-x-auto rounded-md bg-white p-2 text-xs text-gray-700 dark:bg-gray-900 dark:text-gray-300">
          {item.result}
        </pre>
      )}
    </div>
  )
}

function StructuredDisplay({ display, onPreview }: { display: Record<string, unknown>; onPreview?: (content: string, path: string) => void }) {
  const kind = String(display.kind ?? '')
  if (kind === 'command') {
    return (
      <div className="mt-2 space-y-2 text-xs">
        <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <Terminal size={14} />
          <span>{String(display.shell)} exit {String(display.exit_code)}</span>
        </div>
        <OutputBlock title="stdout" value={String(display.stdout ?? '')} />
        {display.stderr != null ? <OutputBlock title="stderr" value={String(display.stderr)} /> : null}
      </div>
    )
  }
  if (kind === 'directory') {
    const entries = Array.isArray(display.entries) ? display.entries : []
    return (
      <div className="mt-2 max-h-56 overflow-auto rounded-md border border-gray-200 bg-white text-xs dark:border-gray-700 dark:bg-gray-900">
        {entries.length === 0 ? (
          <div className="px-2 py-1.5 text-gray-400">(empty)</div>
        ) : (
          entries.map((entry, i) => {
            const row = entry as { name?: string; type?: string }
            return (
              <div key={`${row.name}-${i}`} className="flex items-center gap-2 border-b border-gray-100 px-2 py-1.5 last:border-0 dark:border-gray-800">
                <FileText size={13} className={row.type === 'dir' ? 'text-blue-500' : 'text-gray-400'} />
                <span className="font-mono">{row.name}</span>
                <span className="ml-auto text-[10px] text-gray-400">{row.type}</span>
              </div>
            )
          })
        )}
      </div>
    )
  }
  if (kind === 'file' || kind === 'tool_doc') {
    const filePath = kind === 'file' ? String(display.path ?? '') : String(display.tool_name ?? '')
    return (
      <div className="mt-2 flex items-center gap-2 rounded-md border border-gray-200 bg-white px-2 py-1.5 text-xs text-gray-600 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
        <span className="flex-1">
          {kind === 'file'
            ? `${String(display.action)} ${String(display.path)} (${String(display.chars)} chars)`
            : `已加载工具说明：${String(display.tool_name)} (${String(display.chars)} chars)`}
        </span>
        {(display.action === 'read' || display.action === 'write' || !!display.previewAvailable) && onPreview && (
          <button
            onClick={() => onPreview(String(display.previewContent ?? ''), filePath)}
            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-blue-600 hover:bg-blue-50"
            title="在右侧预览"
          >
            <Eye size={12} />
            预览
          </button>
        )}
      </div>
    )
  }
  if (kind === 'python') {
    return (
      <div className="mt-2 space-y-2 text-xs">
        <pre className="max-h-60 overflow-auto rounded-md bg-gray-950 p-3 text-[11px] text-gray-100">
          {String(display.code ?? '')}
        </pre>
        {display.cancelled ? (
          <div className="text-amber-600 dark:text-amber-300">用户取消执行</div>
        ) : (
          <>
            <div className="text-gray-500 dark:text-gray-400">exit {String(display.exit_code ?? '')}</div>
            <OutputBlock title="stdout" value={String(display.stdout ?? '')} />
            {display.stderr != null ? <OutputBlock title="stderr" value={String(display.stderr)} /> : null}
          </>
        )}
      </div>
    )
  }
  if (kind === 'subagent') {
    return (
      <div className="mt-2 rounded-md border border-gray-200 bg-white p-2 text-xs text-gray-700 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
        <div className="mb-1 flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <BookOpen size={14} />
          <span>{String(display.title ?? display.role ?? 'subagent')}</span>
        </div>
        <div className="line-clamp-2 text-gray-600 dark:text-gray-300">{String(display.task ?? '')}</div>
        {display.result ? (
          <pre className="mt-2 max-h-48 overflow-auto rounded-md bg-gray-50 p-2 text-[11px] dark:bg-gray-950">
            {String(display.result)}
          </pre>
        ) : null}
      </div>
    )
  }
  if (kind === 'json') {
    return (
      <pre className="mt-2 max-h-72 overflow-auto rounded-md border border-gray-200 bg-white p-2 text-[11px] text-gray-700 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
        {JSON.stringify(display.data ?? display, null, 2)}
      </pre>
    )
  }
  return null
}

function OutputBlock({ title, value }: { title: string; value: string }) {
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wide text-gray-400">{title}</div>
      <pre className="max-h-48 overflow-auto rounded-md bg-white p-2 text-[11px] text-gray-700 dark:bg-gray-900 dark:text-gray-300">
        {value || '(empty)'}
      </pre>
    </div>
  )
}
