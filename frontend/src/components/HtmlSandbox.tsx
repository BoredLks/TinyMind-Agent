import { useEffect, useMemo, useRef, useState } from 'react'

interface SandboxError {
  category: string
  message: string
  source?: string
  line?: number
  col?: number
  stack?: string
}

const bridgeScript = `
<script>
(function () {
  'use strict';
  function send(type, payload) {
    parent.postMessage({ type: type, payload: payload }, '*');
  }
  window.onerror = function (message, source, line, col, error) {
    send('sandbox-error', {
      category: 'runtime',
      message: String(message || 'Runtime error'),
      source: source,
      line: line,
      col: col,
      stack: error && error.stack ? error.stack : ''
    });
    return false;
  };
  window.addEventListener('unhandledrejection', function (event) {
    var reason = event.reason || {};
    send('sandbox-error', {
      category: 'unhandled-rejection',
      message: typeof reason === 'string' ? reason : (reason.message || String(reason)),
      stack: reason && reason.stack ? reason.stack : ''
    });
  });
  var originalWrite = document.write.bind(document);
  document.write = function () {
    send('sandbox-error', {
      category: 'document-write',
      message: 'document.write() was called inside the sandbox.'
    });
    return originalWrite.apply(document, arguments);
  };
  function reportHeight() {
    var body = document.body || document.documentElement;
    var doc = document.documentElement;
    var height = Math.max(
      body ? body.scrollHeight : 0,
      body ? body.offsetHeight : 0,
      doc ? doc.scrollHeight : 0,
      doc ? doc.offsetHeight : 0,
      180
    );
    parent.postMessage({ type: 'sandbox-resize', height: height }, '*');
  }
  window.addEventListener('load', reportHeight);
  if (window.ResizeObserver && document.body) {
    new ResizeObserver(reportHeight).observe(document.body);
  }
  [60, 200, 600, 1200, 2400].forEach(function (delay) {
    setTimeout(reportHeight, delay);
  });
})();
</script>`

const baseStyle = `
<style>
  *,
  *::before,
  *::after { box-sizing: border-box; }
  html { min-height: 100%; }
  body {
    margin: 0;
    color: #1f2937;
    background: transparent;
    font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
      "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    overflow-x: hidden;
  }
  .markdown-body { padding: 0; }
  h1, h2, h3, h4, h5, h6 { margin: 14px 0 8px; line-height: 1.3; }
  h1 { font-size: 1.5rem; }
  h2 { font-size: 1.25rem; }
  h3 { font-size: 1.1rem; }
  p, ul, ol, pre, table, blockquote { margin: 8px 0; }
  ul, ol { padding-left: 20px; }
  code { border-radius: 4px; background: #f3f4f6; padding: 2px 5px; font-size: 0.9em; }
  pre { overflow-x: auto; border-radius: 8px; background: #111827; padding: 12px; color: #f9fafb; }
  pre code { background: transparent; padding: 0; color: inherit; }
  table { width: 100%; border-collapse: collapse; }
  th, td { border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; }
  th { background: #f9fafb; }
  img, video, canvas, svg { max-width: 100%; height: auto; }
  blockquote { border-left: 3px solid #6b7280; padding-left: 12px; color: #4b5563; }
  a { color: #2563eb; }
</style>`

function isFullDocument(html: string) {
  const trimmed = html.trim()
  return /^<!doctype html/i.test(trimmed) || /^<html[\s>]/i.test(trimmed)
}

function buildSrcDoc(html: string) {
  if (isFullDocument(html)) {
    let doc = html
    if (/<\/head>/i.test(doc)) {
      doc = doc.replace(/<\/head>/i, `${baseStyle}</head>`)
    } else {
      doc = `${baseStyle}${doc}`
    }
    if (/<\/body>/i.test(doc)) {
      return doc.replace(/<\/body>/i, `${bridgeScript}</body>`)
    }
    return `${doc}${bridgeScript}`
  }
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
${baseStyle}
</head>
<body>
<div class="markdown-body">
${html}
</div>
${bridgeScript}
</body>
</html>`
}

export default function HtmlSandbox({ html }: { html: string }) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [height, setHeight] = useState(220)
  const [loaded, setLoaded] = useState(false)
  const [errors, setErrors] = useState<SandboxError[]>([])
  const srcDoc = useMemo(() => buildSrcDoc(html), [html])

  useEffect(() => {
    setLoaded(false)
    setErrors([])
    setHeight(220)
  }, [html])

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (iframeRef.current?.contentWindow && event.source !== iframeRef.current.contentWindow) return
      if (event.data?.type === 'sandbox-resize' && typeof event.data.height === 'number') {
        setHeight(Math.max(180, Math.ceil(event.data.height)))
        return
      }
      if (event.data?.type === 'sandbox-error' && event.data.payload) {
        const payload = event.data.payload as Partial<SandboxError>
        setErrors((cur) => [
          ...cur,
          {
            category: payload.category || 'unknown',
            message: payload.message || '(no message)',
            source: payload.source,
            line: payload.line,
            col: payload.col,
            stack: payload.stack,
          },
        ])
      }
    }
    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [])

  return (
    <div className="relative my-2 overflow-hidden rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-900">
      <div className="flex h-8 items-center justify-between border-b border-gray-200 bg-gray-50 px-3 text-[11px] text-gray-500 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400">
        <span>HTML sandbox</span>
        <span>{loaded ? `${height}px` : 'loading'}</span>
      </div>
      {!loaded && (
        <div className="flex h-28 items-center justify-center gap-2 text-xs text-gray-400">
          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-700" />
          渲染中
        </div>
      )}
      {errors.length > 0 && (
        <div className="border-b border-red-200 bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
          沙箱内 {errors.length} 个错误：{errors[errors.length - 1].message}
        </div>
      )}
      <iframe
        ref={iframeRef}
        title="sandboxed HTML preview"
        sandbox="allow-scripts allow-modals"
        srcDoc={srcDoc}
        onLoad={() => setLoaded(true)}
        style={{ height }}
        className={loaded ? 'block w-full border-0 bg-white' : 'absolute w-full border-0 opacity-0'}
      />
    </div>
  )
}
