import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import remarkGfm from 'remark-gfm'

import HtmlSandbox from './HtmlSandbox'

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      /* clipboard unavailable */
    }
  }
  return (
    <>
      {lang.toLowerCase() === 'html' && <HtmlSandbox html={code} />}
      <div className="relative my-2">
        <button
          onClick={copy}
          className="absolute right-2 top-2 z-10 rounded bg-gray-700/80 px-2 py-0.5 text-[11px] text-gray-100 hover:bg-gray-600"
        >
          {copied ? '已复制' : '复制'}
        </button>
        <SyntaxHighlighter
          language={lang}
          style={oneDark}
          customStyle={{ margin: 0, borderRadius: '0.5rem', fontSize: '0.8rem', padding: '0.75rem' }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </>
  )
}

export default function Markdown({ content }: { content: string }) {
  return (
    <div className="text-sm leading-relaxed [&_a]:text-blue-600 [&_h1]:my-2 [&_h1]:text-base [&_h1]:font-semibold [&_h2]:my-2 [&_h2]:font-semibold [&_li]:my-0.5 [&_ol]:my-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-1 [&_table]:my-2 [&_td]:border [&_td]:border-gray-300 [&_td]:px-2 [&_th]:border [&_th]:border-gray-300 [&_th]:px-2 [&_ul]:my-1 [&_ul]:list-disc [&_ul]:pl-5">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '')
            const text = String(children)
            const isBlock = Boolean(match) || text.includes('\n')
            if (isBlock) {
              return <CodeBlock lang={match?.[1] ?? 'text'} code={text.replace(/\n$/, '')} />
            }
            return (
              <code
                className="rounded bg-gray-100 px-1 py-0.5 text-[0.85em] dark:bg-gray-700"
                {...props}
              >
                {children}
              </code>
            )
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
