import { useChatStore } from '../stores/chatStore'

export default function StatusIndicator() {
  const status = useChatStore((s) => s.status)
  const error = useChatStore((s) => s.error)
  const retrying = useChatStore((s) => s.retrying)
  const notice = useChatStore((s) => s.notice)

  if (status === 'idle' && retrying == null && !notice) return null

  return (
    <div className="px-5 pb-1 text-xs">
      {retrying != null && <span className="text-amber-500">重试中（第 {retrying} 次）…</span>}
      {retrying == null && status === 'generating' && <span className="text-gray-400">生成中…</span>}
      {status === 'error' && <span className="text-red-500">出错：{error}</span>}
      {notice && <span className="ml-2 text-gray-400">{notice}</span>}
    </div>
  )
}
