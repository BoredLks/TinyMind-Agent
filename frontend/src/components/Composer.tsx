import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type RefCallback } from 'react'
import { Cpu, Send, Square, Image as ImageIcon, X } from 'lucide-react'

import { api } from '../api/restClient'
import { cancelGeneration, sendMessage } from '../api/wsClient'
import { useChatStore } from '../stores/chatStore'
import { useSessionsStore } from '../stores/sessionsStore'
import { useSettingsStore } from '../stores/settingsStore'

interface MentionItem {
  type: string
  name: string
  label: string
  description: string
}

const TYPE_LABELS: Record<string, string> = {
  tool: '🔧 工具',
  skill: '💡 技能',
  file: '📄 文件',
  dir: '📁 目录',
}

interface ImageAttachment {
  id: string
  name: string
  dataUrl: string
  base64: string
}

export default function Composer() {
  const [text, setText] = useState('')
  const [providerId, setProviderId] = useState('')
  const [images, setImages] = useState<ImageAttachment[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const status = useChatStore((s) => s.status)
  const approvalMode = useChatStore((s) => s.approvalMode)
  const setApprovalMode = useChatStore((s) => s.setApprovalMode)
  const settings = useSettingsStore((s) => s.settings)
  const busy = status === 'generating'

  // @ mention state
  const [mentionOpen, setMentionOpen] = useState(false)
  const [mentionQuery, setMentionQuery] = useState('')
  const [mentionItems, setMentionItems] = useState<MentionItem[]>([])
  const [mentionIndex, setMentionIndex] = useState(0)
  const [mentionStart, setMentionStart] = useState(-1)
  const mentionRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (settings?.active_provider_id) setProviderId(settings.active_provider_id)
  }, [settings?.active_provider_id])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${Math.min(180, Math.max(44, el.scrollHeight))}px`
  }, [text])

  // Fetch mention items when query changes
  const fetchMentions = useCallback(async (q: string) => {
    try {
      const res = await api.listMentions(q)
      setMentionItems(res.items)
      setMentionIndex(0)
    } catch {
      setMentionItems([])
    }
  }, [])

  useEffect(() => {
    if (mentionOpen) {
      const timer = setTimeout(() => void fetchMentions(mentionQuery), 150)
      return () => clearTimeout(timer)
    }
  }, [mentionOpen, mentionQuery, fetchMentions])

  // Close mention dropdown when clicking outside
  useEffect(() => {
    if (!mentionOpen) return
    const handler = (e: MouseEvent) => {
      if (mentionRef.current && !mentionRef.current.contains(e.target as Node)) {
        setMentionOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [mentionOpen])

  const handleTextChange = (value: string) => {
    setText(value)

    // Detect @ mentions
    const el = textareaRef.current
    if (!el) return
    const cursor = el.selectionStart ?? value.length
    const before = value.slice(0, cursor)
    const atMatch = before.match(/@(\S{0,30})$/)

    if (atMatch) {
      const start = cursor - atMatch[0].length
      setMentionStart(start)
      setMentionQuery(atMatch[1])
      if (!mentionOpen) setMentionOpen(true)
    } else {
      setMentionOpen(false)
      setMentionStart(-1)
    }
  }

  const insertMention = (item: MentionItem) => {
    if (mentionStart < 0) return
    const before = text.slice(0, mentionStart)
    const el = textareaRef.current
    const cursor = el?.selectionStart ?? text.length
    const after = text.slice(cursor)
    const tag = `@${item.name} `
    const newText = before + tag + after
    setText(newText)
    setMentionOpen(false)
    setMentionStart(-1)

    // Restore focus and cursor
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus()
        const pos = before.length + tag.length
        textareaRef.current.setSelectionRange(pos, pos)
      }
    }, 0)
  }

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return
    Array.from(files).forEach((file) => {
      if (!file.type.startsWith('image/')) return
      const reader = new FileReader()
      reader.onload = () => {
        const dataUrl = reader.result as string
        const base64 = dataUrl.split(',')[1] || ''
        setImages((prev) => [...prev, { id: crypto.randomUUID(), name: file.name, dataUrl, base64 }])
      }
      reader.readAsDataURL(file)
    })
    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const removeImage = (id: string) => {
    setImages((prev) => prev.filter((img) => img.id !== id))
  }

  const submit = async () => {
    const content = text.trim()
    if ((!content && images.length === 0) || busy) return
    if (!useSessionsStore.getState().currentId) {
      await useSessionsStore.getState().create()
    }

    let messageContent = content
    if (images.length > 0) {
      // Format: text + image data for multimodal models
      const imageData = images.map((img) => ({
        type: 'image_url',
        image_url: { url: img.dataUrl },
      }))
      messageContent = JSON.stringify({
        type: 'multimodal',
        text: content,
        images: imageData,
      })
    }

    void sendMessage(messageContent, providerId || undefined)
    setText('')
    setImages([])
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (mentionOpen && mentionItems.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setMentionIndex((i) => (i + 1) % mentionItems.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setMentionIndex((i) => (i - 1 + mentionItems.length) % mentionItems.length)
        return
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault()
        insertMention(mentionItems[mentionIndex])
        return
      }
      if (e.key === 'Escape') {
        setMentionOpen(false)
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void submit()
    }
  }

  // Ref callback that scrolls the selected item into view
  const selectedRef: RefCallback<HTMLButtonElement> = useCallback(
    (node) => {
      if (node) {
        node.scrollIntoView({ block: 'nearest' })
      }
    },
    [mentionIndex],
  )

  return (
    <div className="relative border-t border-gray-200 bg-white/95 p-3 dark:border-gray-700 dark:bg-gray-900/95" style={{ zIndex: 20 }}>
      {/* @ Mention dropdown */}
      {mentionOpen && mentionItems.length > 0 && (
        <div
          ref={mentionRef}
          className="absolute bottom-full left-3 right-3 mb-1 max-h-60 overflow-auto rounded-xl border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-800"
        >
          <div className="px-3 py-1.5 text-[11px] text-gray-400">
            提及（↑↓ 选择，Enter 插入）
          </div>
          {mentionItems.map((item, i) => (
            <button
              key={`${item.type}-${item.name}`}
              ref={i === mentionIndex ? selectedRef : undefined}
              className={`flex w-full items-start gap-2 px-3 py-2 text-left text-xs ${
                i === mentionIndex
                  ? 'bg-blue-50 dark:bg-blue-900/30'
                  : 'hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
              onMouseDown={(e) => {
                e.preventDefault()
                insertMention(item)
              }}
              onMouseEnter={() => setMentionIndex(i)}
            >
              <span className="shrink-0 rounded bg-gray-100 px-1.5 py-0.5 text-[10px] dark:bg-gray-700">
                {TYPE_LABELS[item.type] ?? item.type}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium text-gray-900 dark:text-gray-100">
                  {item.name}
                </span>
                <span className="block truncate text-gray-500 dark:text-gray-400">
                  {item.description}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}

      {/* Image preview area */}
      {images.length > 0 && (
        <div className="mx-auto mb-2 flex max-w-4xl flex-wrap gap-2">
          {images.map((img) => (
            <div key={img.id} className="group relative">
              <img
                src={img.dataUrl}
                alt={img.name}
                className="h-20 w-20 rounded-lg border border-gray-200 object-cover dark:border-gray-600"
              />
              <button
                onClick={() => removeImage(img.id)}
                className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-white opacity-0 transition-opacity group-hover:opacity-100"
                title="移除图片"
              >
                <X size={12} />
              </button>
              <div className="absolute bottom-0 left-0 right-0 truncate rounded-b-lg bg-black/50 px-1 py-0.5 text-[9px] text-white">
                {img.name}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mx-auto max-w-4xl rounded-lg border border-gray-200 bg-gray-50 p-2 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => handleTextChange(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          aria-label="输入消息"
          placeholder="输入消息（@ 可提及工具/技能/文件）"
          className="max-h-44 min-h-11 w-full resize-none bg-transparent px-2 py-2 text-sm leading-6 focus:outline-none dark:text-gray-100"
        />
        <div className="flex items-center justify-between gap-2 border-t border-gray-200 pt-2 dark:border-gray-700">
          <div className="flex min-w-0 items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <label className="flex items-center gap-1.5">
              <Cpu size={14} />
              <select
                aria-label="对话模型提供商"
                className="max-w-56 truncate rounded-md border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-600 dark:bg-gray-900"
                value={providerId}
                onChange={(e) => setProviderId(e.target.value)}
              >
                {(settings?.providers ?? []).filter((p) => p.enabled).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label} / {p.model}
                  </option>
                ))}
              </select>
            </label>
            <div className="flex items-center gap-1 rounded-md border border-gray-300 bg-white px-1.5 py-0.5 dark:border-gray-600 dark:bg-gray-900" role="radiogroup" aria-label="审批模式">
              <button
                onClick={() => setApprovalMode('auto_approve')}
                className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
                  approvalMode === 'auto_approve'
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300'
                    : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'
                }`}
                title="全部同意：工具调用自动批准"
              >
                全部同意
              </button>
              <button
                onClick={() => setApprovalMode('require_approval')}
                className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
                  approvalMode === 'require_approval'
                    ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'
                    : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'
                }`}
                title="需要批准：工具调用需手动审批"
              >
                需要批准
              </button>
            </div>
          </div>
          {busy ? (
            <button
              onClick={() => cancelGeneration()}
              className="inline-flex h-8 items-center gap-1.5 rounded-md bg-red-600 px-3 text-sm font-medium text-white"
              title="停止生成"
            >
              <Square size={14} fill="currentColor" />
              停止
            </button>
          ) : (
            <div className="flex items-center gap-1.5">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleImageSelect}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex h-8 w-8 items-center justify-center rounded-md text-gray-400 hover:bg-gray-200 hover:text-gray-600 dark:hover:bg-gray-700"
                title="添加图片"
              >
                <ImageIcon size={18} />
              </button>
              <button
                onClick={submit}
                disabled={!text.trim() && images.length === 0}
                className="inline-flex h-8 items-center gap-1.5 rounded-md bg-gray-900 px-3 text-sm font-medium text-white disabled:opacity-40 dark:bg-gray-100 dark:text-gray-900"
                title="发送"
              >
                <Send size={14} />
                发送
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}