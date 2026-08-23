import { useState } from 'react'
import { Copy, Check, FileCode2, FileText, Globe, Image, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

import HtmlSandbox from './HtmlSandbox'
import { usePreviewStore } from '../stores/previewStore'

function looksLikeHtml(content: string) {
  const trimmed = content.trim()
  return /^<!doctype html/i.test(trimmed) || /^<html[\s>]/i.test(trimmed)
}

function looksLikeJson(content: string): boolean {
  const t = content.trim()
  return (t.startsWith('{') && t.endsWith('}')) || (t.startsWith('[') && t.endsWith(']'))
}

function looksLikeImageBase64(content: string): boolean {
  return content.startsWith('data:image/')
}

function extToLang(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase() ?? ''
  const map: Record<string, string> = {
    js: 'javascript', jsx: 'jsx', ts: 'typescript', tsx: 'tsx',
    py: 'python', rb: 'ruby', java: 'java', c: 'c', cpp: 'cpp',
    h: 'c', cs: 'csharp', go: 'go', rs: 'rust', php: 'php',
    sh: 'bash', bat: 'batch', ps1: 'powershell', sql: 'sql',
    css: 'css', html: 'html', json: 'json', yaml: 'yaml', yml: 'yaml',
    toml: 'toml', xml: 'xml', md: 'markdown', txt: 'text',
    svg: 'xml', graphql: 'graphql', vue: 'html', svelte: 'html',
  }
  return map[ext] ?? 'text'
}

function isCodeFile(path: string): boolean {
  const codeExts = ['.js', '.ts', '.tsx', '.jsx', '.py', '.java', '.c', '.cpp', '.h',
    '.cs', '.go', '.rs', '.php', '.rb', '.css', '.scss', '.less',
    '.json', '.yaml', '.yml', '.toml', '.xml', '.svg',
    '.sh', '.bat', '.ps1', '.sql', '.graphql', '.vue', '.svelte',
    '.r', '.swift', '.kt', '.scala', '.lua', '.vim', '.zsh', '.fish']
  const lower = path.toLowerCase()
  return codeExts.some((ext) => lower.endsWith(ext))
}

function isImageFile(path: string): boolean {
  const lower = path.toLowerCase()
  return ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.ico'].some((ext) => lower.endsWith(ext))
}

function isMarkdownFile(path: string): boolean {
  return path.toLowerCase().endsWith('.md')
}

function isHtmlFile(path: string): boolean {
  const lower = path.toLowerCase()
  return lower.endsWith('.html') || lower.endsWith('.htm')
}

type PreviewKind = 'html' | 'markdown' | 'code' | 'image' | 'json' | 'text'

function detectKind(path: string, content: string): PreviewKind {
  if (isHtmlFile(path) || looksLikeHtml(content)) return 'html'
  if (isImageFile(path) || looksLikeImageBase64(content)) return 'image'
  if (isMarkdownFile(path)) return 'markdown'
  if (isCodeFile(path)) return 'code'
  if (looksLikeJson(content) && content.length < 50000) return 'json'
  if (extToLang(path) !== 'text') return 'code'
  return 'text'
}

function getKindIcon(kind: PreviewKind) {
  switch (kind) {
    case 'html': return <Globe size={13} className="text-orange-500" />
    case 'markdown': return <FileText size={13} className="text-blue-500" />
    case 'code': return <FileCode2 size={13} className="text-green-500" />
    case 'image': return <Image size={13} className="text-purple-500" />
    case 'json': return <FileCode2 size={13} className="text-yellow-500" />
    default: return <FileText size={13} className="text-gray-400" />
  }
}

function getKindLabel(kind: PreviewKind): string {
  switch (kind) {
    case 'html': return 'HTML 预览'
    case 'markdown': return 'Markdown 预览'
    case 'code': return '代码预览'
    case 'image': return '图片预览'
    case 'json': return 'JSON 预览'
    default: return '文本预览'
  }
}

export default function FilePreviewPanel({ width }: { width?: number } = {}) {
  const file = usePreviewStore((s) => s.file)
  const close = usePreviewStore((s) => s.close)
  const [copied, setCopied] = useState(false)

  if (!file) return null

  const kind = detectKind(file.path, file.content)

  const copyContent = async () => {
    try {
      await navigator.clipboard.writeText(file.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* ignore */ }
  }

  return (
    <div className="w-full max-w-lg border-l border-gray-200 bg-white flex flex-col shrink-0 overflow-hidden" style={{ minWidth: 360, width: width ?? undefined, maxWidth: width ?? undefined }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 bg-gray-50/80 shrink-0">
        <div className="min-w-0 flex-1 flex items-center gap-2">
          {getKindIcon(kind)}
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium truncate">{file.path}</p>
            <p className="text-[10px] text-gray-400">{getKindLabel(kind)} · {file.content.length} 字符</p>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={copyContent}
            className="p-1.5 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-600"
            title="复制内容"
          >
            {copied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
          </button>
          <button
            onClick={close}
            className="p-1.5 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-600"
            title="关闭预览"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {kind === 'html' ? (
          <HtmlSandbox html={file.content} />
        ) : kind === 'image' ? (
          <div className="flex items-center justify-center p-4 h-full">
            <img
              src={looksLikeImageBase64(file.content) ? file.content : `data:image/png;base64,${file.content}`}
              alt={file.path}
              className="max-w-full max-h-full object-contain rounded-lg border border-gray-200"
            />
          </div>
        ) : kind === 'markdown' ? (
          <div className="p-4 text-sm leading-relaxed prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{file.content}</ReactMarkdown>
          </div>
        ) : kind === 'json' ? (
          <div className="p-4">
            <SyntaxHighlighter
              language="json"
              style={oneDark}
              customStyle={{ margin: 0, borderRadius: '0.5rem', fontSize: '0.8rem', padding: '1rem' }}
              showLineNumbers
            >
              {file.content}
            </SyntaxHighlighter>
          </div>
        ) : kind === 'code' ? (
          <SyntaxHighlighter
            language={extToLang(file.path)}
            style={oneDark}
            customStyle={{ margin: 0, borderRadius: 0, fontSize: '0.8rem', padding: '1rem', minHeight: '100%' }}
            showLineNumbers
          >
            {file.content}
          </SyntaxHighlighter>
        ) : (
          <pre className="p-4 text-xs leading-relaxed text-gray-700 whitespace-pre-wrap break-words font-mono">
            {file.content}
          </pre>
        )}
      </div>
    </div>
  )
}