import { useEffect, useRef, useState } from 'react'
import { Bot, Check, CheckCircle2, ChevronDown, ChevronUp, Copy, Loader2, User } from 'lucide-react'

import { useChatStore } from '../stores/chatStore'
import Markdown from './Markdown'
import HtmlSandbox from './HtmlSandbox'
import ToolCard from './ToolCard'

function looksLikeHtml(content: string) {
  const trimmed = content.trim()
  return /^<!doctype html/i.test(trimmed) || /^<html[\s>]/i.test(trimmed) || /<body[\s>]/i.test(trimmed)
}

function renderMessageContent(isUser: boolean, content: string) {
  if (!content) return <span className="text-gray-400">…</span>

  // Check if content is multimodal (JSON with images)
  try {
    const parsed = JSON.parse(content)
    if (parsed.type === 'multimodal' && Array.isArray(parsed.images)) {
      return (
        <>
          {parsed.text && <span>{parsed.text}</span>}
          <div className="mt-2 flex flex-wrap gap-2">
            {parsed.images.map((img: { image_url: { url: string } }, i: number) => (
              <img
                key={i}
                src={img.image_url.url}
                alt={`附件 ${i + 1}`}
                className="max-h-60 max-w-xs rounded-lg border border-gray-200 object-contain dark:border-gray-600"
              />
            ))}
          </div>
        </>
      )
    }
  } catch {
    // Not JSON, proceed as normal
  }

  if (isUser) return content
  if (looksLikeHtml(content)) return <HtmlSandbox html={content} />
  return <Markdown content={content} />
}

function ThinkingBlock({ reasoning, isStillThinking }: { reasoning: string; isStillThinking: boolean }) {
  const [open, setOpen] = useState(false)
  if (!reasoning) return null
  const charCount = reasoning.length
  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[11px] text-gray-400 hover:text-gray-600 transition-colors"
      >
        {isStillThinking ? (
          <Loader2 size={12} className="animate-spin text-blue-400" />
        ) : (
          <CheckCircle2 size={12} className="text-green-500" />
        )}
        <span className="font-medium">{isStillThinking ? '思考中...' : '思考完成'}</span>
        <span className="text-[10px] text-gray-300">({charCount} 字符)</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && (
        <pre className="mt-1.5 max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-gray-50 border border-gray-100 p-3 text-[11px] leading-relaxed text-gray-500">
          {reasoning}
        </pre>
      )}
    </div>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* fallback: ignore */ }
  }
  return (
    <button
      onClick={() => void handleCopy()}
      className="absolute -right-1 -top-1 rounded-md bg-gray-200 p-1 text-gray-500 opacity-0 transition-opacity hover:bg-gray-300 hover:text-gray-700 group-hover:opacity-100 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600"
      title="复制内容"
    >
      {copied ? <Check size={12} className="text-green-500" /> : <Copy size={12} />}
    </button>
  )
}

export default function MessageList() {
  const items = useChatStore((s) => s.items)
  const reasoningActive = useChatStore((s) => s.reasoningActive)
  const status = useChatStore((s) => s.status)
  const endRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) return
    const { scrollTop, scrollHeight, clientHeight } = container
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 150
    // Only auto-scroll if user is near the bottom OR if it's a new user message
    if (isNearBottom) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [items])

  if (items.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center px-5 text-center">
        <div className="space-y-2">
          <div className="text-2xl font-semibold tracking-tight text-gray-900 dark:text-gray-100">
            SuperAgent
          </div>
          <div className="text-sm text-gray-400">Local project agent</div>
        </div>
      </div>
    )
  }

  return (
    <div ref={scrollContainerRef} className="flex-1 space-y-2 overflow-y-auto px-5 py-6">
      {items.map((it) => {
        if (it.kind === 'tool') {
          // Hide completed subagent tool cards — result is shown in assistant's response
          if (it.subagent && (it.state === 'ok' || it.state === 'error')) return null
          return <ToolCard key={it.id} item={it} />
        }
        const user = it.role === 'user'
        return (
          <div key={it.id} className={user ? 'flex justify-end' : 'flex justify-start'}>
            <div
              className={
                'group relative ' +
                (user
                  ? 'my-1 max-w-[76%] rounded-xl rounded-br-md border border-gray-300 bg-white px-4 py-2.5 text-sm leading-relaxed text-gray-900 shadow-sm dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100'
                  : 'my-1 max-w-[82%] rounded-xl rounded-bl-md border border-gray-200 bg-gray-50 px-4 py-2.5 text-sm leading-relaxed text-gray-900 shadow-sm dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100')
              }
            >
              <CopyButton text={it.content} />
              <div className="mb-1 flex items-center gap-1.5 text-[11px] font-medium uppercase text-gray-400">
                {user ? <User size={12} /> : <Bot size={12} />}
                {user ? 'User' : 'Assistant'}
              </div>
              {'reasoning' in it && (it as { reasoning?: string }).reasoning && <ThinkingBlock reasoning={(it as { reasoning?: string }).reasoning!} isStillThinking={reasoningActive && status === 'generating' && it === items[items.length - 1]} />}
              {renderMessageContent(user, it.content)}
            </div>
          </div>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}