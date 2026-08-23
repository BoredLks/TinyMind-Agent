import { useEffect, useState } from 'react'
import {
  Brain, ChevronDown, ChevronRight, Cpu, Folder, FolderOpen, MessageSquare,
  Pencil, Plus, Puzzle, Trash2, Wrench, X,
} from 'lucide-react'

import { api, type SessionMeta } from './api/restClient'
import { checkBackendHealth } from './api/wsClient'
import ChatView from './components/ChatView'
import FilePreviewPanel from './components/FilePreviewPanel'
import ResizableDivider from './components/ResizableDivider'
import MemoryPanel from './components/MemoryPanel'
import SettingsPanel from './components/SettingsPanel'
import SkillsPanel from './components/SkillsPanel'
import ToolsPanel from './components/ToolsPanel'
import StageBar from './components/StageBar'
import { useChatStore } from './stores/chatStore'
import { useProjectsStore } from './stores/projectsStore'
import { useSessionsStore } from './stores/sessionsStore'
import { useSettingsStore } from './stores/settingsStore'
import { usePreviewStore } from './stores/previewStore'
import { useSkillsStore } from './stores/skillsStore'
import { useToolsStore } from './stores/toolsStore'

type NavTab = 'chat' | 'memory' | 'models'

function parseSessionMeta(metaJson?: string | null): Record<string, unknown> {
  if (!metaJson) return {}
  try {
    const parsed = JSON.parse(metaJson)
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

export default function App() {
  const bootstrap = useSessionsStore((s) => s.bootstrap)
  const undoLastTurn = useSessionsStore((s) => s.undoLastTurn)
  const loadProjects = useProjectsStore((s) => s.load)
  const projects = useProjectsStore((s) => s.projects)
  const addProject = useProjectsStore((s) => s.add)
  const removeProject = useProjectsStore((s) => s.remove)
  const renameProject = useProjectsStore((s) => s.rename)
  const projectsError = useProjectsStore((s) => s.error)
  const generating = useChatStore((s) => s.generating)
  const sessions = useSessionsStore((s) => s.sessions)
  const currentId = useSessionsStore((s) => s.currentId)
  const loadSettings = useSettingsStore((s) => s.load)
  const openSettings = useSettingsStore((s) => s.setOpen)
  const openSkills = useSkillsStore((s) => s.setOpen)
  const openTools = useToolsStore((s) => s.setOpen)
  const toolCount = useToolsStore((s) => s.tools.length)
  const loadTools = useToolsStore((s) => s.load)
  const wsConnected = useChatStore((s) => s.wsConnected)
  const previewFile = usePreviewStore((s) => s.file)
  const usage = useChatStore((s) => s.usage)
  const status = useChatStore((s) => s.status)
  const settings = useSettingsStore((s) => s.settings)
  const memoryEnabled = settings?.memory_enabled ?? true
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [balance, setBalance] = useState<{ balance: string; currency: string } | null>(null)
  const [balanceLoading, setBalanceLoading] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(256)
  const [previewWidth, setPreviewWidth] = useState(384)

  const [nav, setNav] = useState<NavTab>('chat')
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set())
  const [projectDialog, setProjectDialog] = useState<{ path: string; name: string } | null>(null)
  const [textDialog, setTextDialog] = useState<
    | { kind: 'renameSession'; title: string; label: string; initial: string; id: string }
    | { kind: 'renameProject'; title: string; label: string; initial: string; id: string }
    | null
  >(null)
  const [textValue, setTextValue] = useState('')
  const [confirmDialog, setConfirmDialog] = useState<
    | { kind: 'deleteSession'; title: string; message: string; id: string }
    | { kind: 'deleteProject'; title: string; message: string; id: string }
    | null
  >(null)

  const select = useSessionsStore((s) => s.select)
  const create = useSessionsStore((s) => s.create)
  const rename = useSessionsStore((s) => s.rename)
  const remove = useSessionsStore((s) => s.remove)

  const toggleProject = (id: string) =>
    setExpandedProjects((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  const expandProject = (id: string) => setExpandedProjects((prev) => new Set(prev).add(id))

  // Add a project: try the desktop's native folder dialog first, fall back to
  // a manual path-entry dialog (e.g. when running in a dev browser).
  const browseAndAddProject = async () => {
    try {
      const res = await api.pickFolder()
      if (res.available) {
        if (res.path) {
          await addProject(res.path)
          const newId = useProjectsStore.getState().currentId
          if (newId && !useProjectsStore.getState().error) expandProject(newId)
        }
        return
      }
    } catch {
      // fall through to manual entry
    }
    setProjectDialog({ path: '', name: '' })
  }
  const submitProjectDialog = async () => {
    if (!projectDialog) return
    const path = projectDialog.path.trim()
    if (!path) return
    await addProject(path, projectDialog.name.trim() || undefined)
    if (!useProjectsStore.getState().error) {
      const newId = useProjectsStore.getState().currentId
      if (newId) expandProject(newId)
      setProjectDialog(null)
    }
  }
  const newSessionInProject = (projectId: string) => {
    setNav('chat')
    expandProject(projectId)
    void create(projectId)
  }

  useEffect(() => {
    void loadProjects()
    void bootstrap()
    void loadSettings()
    void loadTools()
    void checkBackendHealth()
    const interval = setInterval(() => void checkBackendHealth(), 15000)
    return () => clearInterval(interval)
  }, [bootstrap, loadProjects, loadSettings])

  // Apply the light/dark theme by toggling the `dark` class on <html>
  // (Tailwind darkMode: 'class'). Without this the theme selector did nothing.
  useEffect(() => {
    document.documentElement.classList.toggle('dark', settings?.theme === 'dark')
  }, [settings?.theme])

  // Check DeepSeek balance if a DeepSeek provider is configured
  const deepseekProvider = settings?.providers?.find(
    (p) => (p.id + p.label).toLowerCase().includes('deepseek') && p.enabled && p.has_api_key
  )

  useEffect(() => {
    if (!settings || !deepseekProvider) { setBalance(null); return }
    let cancelled = false
    const check = async () => {
      setBalanceLoading(true)
      try {
        const res = await fetch('/api/balance/deepseek')
        if (res.ok) {
          const d = await res.json()
          // DeepSeek API returns: { balance_infos: [{ total_balance: "9.97", currency: "CNY" }] }
          const info = d.balance_infos?.[0]
          const bal = info?.total_balance ?? d.balance ?? d.data?.balance ?? null
          const cur = info?.currency ?? d.currency ?? 'CNY'
          if (!cancelled && bal !== null) {
            setBalance({ balance: String(bal), currency: String(cur) })
          } else if (!cancelled) {
            setBalance(null)
          }
        } else {
          if (!cancelled) setBalance(null)
        }
      } catch {
        if (!cancelled) setBalance(null)
      } finally {
        if (!cancelled) setBalanceLoading(false)
      }
    }
    void check()
    const balInterval = setInterval(() => void check(), 60000)
    return () => { cancelled = true; clearInterval(balInterval) }
  }, [settings, !!deepseekProvider])

  const hasApiKey = settings?.has_api_key ?? false

  const openTextDialog = (dialog: NonNullable<typeof textDialog>) => {
    setTextDialog(dialog)
    setTextValue(dialog.initial)
  }
  const closeTextDialog = () => { setTextDialog(null); setTextValue('') }
  const submitTextDialog = async () => {
    const value = textValue.trim()
    if (!textDialog || !value) return
    if (textDialog.kind === 'renameSession') await rename(textDialog.id, value)
    else if (textDialog.kind === 'renameProject') await renameProject(textDialog.id, value)
    closeTextDialog()
  }
  const submitConfirmDialog = async () => {
    if (!confirmDialog) return
    if (confirmDialog.kind === 'deleteSession') await remove(confirmDialog.id)
    else if (confirmDialog.kind === 'deleteProject') await removeProject(confirmDialog.id)
    setConfirmDialog(null)
  }

  // Stable "Session #N" numbering across regular (non-subagent) sessions.
  const regulars = sessions.filter((s) => parseSessionMeta(s.meta_json).is_subagent !== true)
  const sessionNumber = new Map(regulars.map((s, idx) => [s.id, regulars.length - idx]))
  const ungroupedSessions = sessions.filter((s) => !s.project_id)
  const sessionsOf = (projectId: string) => sessions.filter((s) => s.project_id === projectId)

  const renderSession = (s: SessionMeta) => {
    const active = s.id === currentId
    const isSub = parseSessionMeta(s.meta_json).is_subagent === true
    const busy = Boolean(generating[s.id])
    return (
      <div
        key={s.id}
        onClick={() => { setNav('chat'); void select(s.id) }}
        className={`group flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs cursor-pointer mb-0.5 ${
          active ? 'bg-gray-200/80 font-medium text-gray-900 dark:bg-gray-700/80 dark:text-gray-100' : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
        }`}
      >
        {isSub && <span className="text-[9px] text-purple-500">↳</span>}
        {busy && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500 animate-pulse" title="生成中" />}
        <span className="truncate flex-1">
          {isSub ? s.title : `Session #${sessionNumber.get(s.id) ?? ''}`}
        </span>
        {!isSub && (
          <span className="flex gap-1 opacity-0 group-hover:opacity-100 shrink-0">
            <button
              aria-label="重命名"
              onClick={(e) => { e.stopPropagation(); openTextDialog({ kind: 'renameSession', title: '重命名会话', label: '名称', initial: s.title, id: s.id }) }}
              className="p-0.5 hover:text-gray-900"
            >
              <Pencil size={11} />
            </button>
            <button
              aria-label="删除"
              onClick={(e) => { e.stopPropagation(); setConfirmDialog({ kind: 'deleteSession', title: '删除会话', message: `确认删除 "${s.title}"？`, id: s.id }) }}
              className="p-0.5 hover:text-red-500"
            >
              <Trash2 size={11} />
            </button>
          </span>
        )}
      </div>
    )
  }

  return (
    <div className="h-screen flex bg-white text-gray-900 overflow-hidden dark:bg-gray-900 dark:text-gray-100">
      {/* ===== LEFT SIDEBAR ===== */}
      <aside className="flex flex-col bg-gray-50/80 border-r border-gray-200 select-none shrink-0 dark:bg-gray-950/40 dark:border-gray-800" style={{ width: sidebarWidth }}>
        {/* Logo */}
        <div className="px-4 pt-4 pb-2">
          <h1 className="text-lg font-bold tracking-tight text-gray-900 dark:text-gray-100">SuperAgent</h1>
        </div>

        {/* Nav Tabs */}
        <nav className="px-3 pb-2 flex gap-1">
          {([['chat', MessageSquare, 'Chatting'], ['memory', Brain, 'Memory'], ['models', Cpu, 'Models']] as const).map(([key, Icon, label]) => (
            <button
              key={key}
              onClick={() => setNav(key)}
              className={`flex-1 flex flex-col items-center gap-0.5 rounded-lg py-1.5 text-[10px] font-medium transition-colors ${
                nav === key
                  ? 'bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900'
                  : 'text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'
              }`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </nav>

        {/* Projects + Session List */}
        <div className="flex-1 overflow-y-auto px-3 pb-2">
          {/* ===== Projects ===== */}
          <div className="flex items-center justify-between mb-1.5 px-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">Projects</span>
            <button
              onClick={() => void browseAndAddProject()}
              className="p-0.5 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-700"
              title="添加项目（选择目录）"
            >
              <Plus size={14} />
            </button>
          </div>
          {projectsError && (
            <div className="mb-1 px-1 text-[10px] text-red-500">{projectsError}</div>
          )}
          {projects.length === 0 && (
            <div className="mb-2 px-1 text-[11px] text-gray-400">暂无项目，点击 + 选择目录</div>
          )}
          {projects.map((p) => {
            const expanded = expandedProjects.has(p.id)
            const projSessions = sessionsOf(p.id)
            return (
              <div key={p.id} className="mb-1">
                <div
                  onClick={() => toggleProject(p.id)}
                  className="group flex items-center gap-1.5 rounded-lg px-1.5 py-1.5 text-xs cursor-pointer text-gray-700 hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-800"
                  title={p.path}
                >
                  {expanded ? <ChevronDown size={13} className="shrink-0 text-gray-400" /> : <ChevronRight size={13} className="shrink-0 text-gray-400" />}
                  {expanded ? <FolderOpen size={13} className="shrink-0 text-amber-500" /> : <Folder size={13} className="shrink-0 text-amber-500" />}
                  <span className="truncate flex-1 font-medium">{p.name}</span>
                  <span className="text-[10px] text-gray-400 shrink-0">{projSessions.length}</span>
                  <span className="flex gap-1 opacity-0 group-hover:opacity-100 shrink-0">
                    <button
                      aria-label="新建对话"
                      onClick={(e) => { e.stopPropagation(); newSessionInProject(p.id) }}
                      className="p-0.5 hover:text-gray-900"
                      title="在该项目下新建对话"
                    >
                      <Plus size={11} />
                    </button>
                    <button
                      aria-label="重命名项目"
                      onClick={(e) => { e.stopPropagation(); openTextDialog({ kind: 'renameProject', title: '重命名项目', label: '名称', initial: p.name, id: p.id }) }}
                      className="p-0.5 hover:text-gray-900"
                    >
                      <Pencil size={11} />
                    </button>
                    <button
                      aria-label="移除项目"
                      onClick={(e) => { e.stopPropagation(); setConfirmDialog({ kind: 'deleteProject', title: '移除项目', message: `移除项目 "${p.name}"？其对话会保留并移到未分组。`, id: p.id }) }}
                      className="p-0.5 hover:text-red-500"
                    >
                      <Trash2 size={11} />
                    </button>
                  </span>
                </div>
                {expanded && (
                  <div className="ml-2.5 mt-0.5 border-l border-gray-200 pl-1.5">
                    {projSessions.length === 0 ? (
                      <div className="px-2 py-1 text-[11px] text-gray-400">无对话，点击项目右侧 + 新建</div>
                    ) : (
                      projSessions.map(renderSession)
                    )}
                  </div>
                )}
              </div>
            )
          })}

          {/* ===== Ungrouped Sessions ===== */}
          <div className="flex items-center justify-between mb-1.5 mt-3 px-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">Sessions</span>
            <button
              onClick={() => void create()}
              className="p-0.5 rounded hover:bg-gray-200 text-gray-400 hover:text-gray-700"
              title="新会话（不属于任何项目）"
            >
              <Plus size={14} />
            </button>
          </div>
          {ungroupedSessions.length === 0 && (
            <div className="px-1 text-[11px] text-gray-400">无未分组对话</div>
          )}
          {ungroupedSessions.map(renderSession)}
        </div>

        {/* System Status Section */}
        <div className="border-t border-gray-200 px-3 py-2 space-y-1 dark:border-gray-800">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-1.5 px-1">System</div>
          <button
            onClick={() => openSettings(true)}
            className="w-full flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
          >
            <Cpu size={13} className={hasApiKey ? 'text-green-500' : 'text-gray-400'} />
            <span className="flex-1 text-left">LLM</span>
            <span className="text-[10px] text-gray-400">{settings?.providers?.filter(p => p.enabled).length ?? 0} 提供商</span>
          </button>
          <button
            onClick={() => { setNav('memory'); setMemoryOpen(true) }}
            className="w-full flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
          >
            <Brain size={13} className={memoryEnabled ? 'text-blue-500' : 'text-gray-400'} />
            <span className="flex-1 text-left">Memory</span>
            <span className="text-[10px] text-gray-400">{memoryEnabled ? 'ON' : 'OFF'}</span>
          </button>
          <button
            onClick={() => openTools(true)}
            className="w-full flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
          >
            <Wrench size={13} className="text-gray-400" />
            <span className="flex-1 text-left">Tools</span>
            <span className="text-[10px] text-gray-400">{toolCount || '...'}</span>
          </button>
          <button
            onClick={() => openSkills(true)}
            className="w-full flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
          >
            <Puzzle size={13} className="text-purple-400" />
            <span className="flex-1 text-left">Skills</span>
            <span className="text-[10px] text-gray-400">{settings?.providers ? '16' : '...'}</span>
          </button>
        </div>
      </aside>

      {/* Sidebar resize handle */}
      <ResizableDivider currentWidth={sidebarWidth} onResize={setSidebarWidth} minWidth={180} maxWidth={400} />

      {/* ===== MAIN CONTENT ===== */}
      <div className="flex-1 flex min-w-0 bg-white dark:bg-gray-900">
        <div className="flex-1 flex flex-col min-w-0">
        {/* Status Strip */}
        <div className="h-7 flex items-center gap-3 px-4 border-b border-gray-100 bg-gray-50/50 text-[10px] text-gray-500 shrink-0 dark:border-gray-800 dark:bg-gray-950/40 dark:text-gray-400">
          <span className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-400'}`} />
            {wsConnected ? 'API 已连接' : 'API 未连接'}
          </span>
          <span className="text-gray-300">|</span>
          <span className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${memoryEnabled ? 'bg-blue-500' : 'bg-gray-300'}`} />
            Memory {memoryEnabled ? 'ON' : 'OFF'}
          </span>
          {balance && (
            <>
              <span className="text-gray-300">|</span>
              <span className="flex items-center gap-1" title="DeepSeek 余额（仅 DeepSeek API 支持）">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                {balance.balance} {balance.currency}
              </span>
            </>
          )}
          {!balance && !balanceLoading && deepseekProvider && (
            <>
              <span className="text-gray-300">|</span>
              <span className="text-gray-400">余额不可用</span>
            </>
          )}
          {usage && (
            <>
              <span className="text-gray-300">|</span>
              <span>{usage.estimated ? '≈' : ''}{usage.session_total} tokens</span>
            </>
          )}
          {status === 'generating' && (
            <>
              <span className="text-gray-300">|</span>
              <span className="text-blue-500 animate-pulse">生成中...</span>
            </>
          )}
          <div className="flex-1" />
          <button
            onClick={() => void undoLastTurn()}
            className="hover:text-gray-700"
            title="撤回最近一轮"
          >
            撤回
          </button>
        </div>

        {/* Content Area */}
        {nav === 'chat' && (
          <div className="flex-1 flex flex-col min-h-0">
            <StageBar />
            <ChatView />
          </div>
        )}

        {nav === 'memory' && (
          <div className="flex-1 overflow-y-auto p-6">
            <MemoryView />
          </div>
        )}

        {nav === 'models' && (
          <div className="flex-1 overflow-y-auto p-6">
            <ModelsView />
          </div>
        )}
        </div>
        {previewFile && <ResizableDivider currentWidth={previewWidth} onResize={setPreviewWidth} minWidth={280} maxWidth={800} invert />}
        {previewFile && <FilePreviewPanel width={previewWidth} />}
      </div>

      {/* Panels (overlays) */}
      <SettingsPanel />
      <SkillsPanel />
      <ToolsPanel />
      <MemoryPanel open={memoryOpen} onClose={() => setMemoryOpen(false)} />

      {/* Add Project Dialog (manual path entry; folder browse used in desktop app) */}
      {projectDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4">
          <div className="w-full max-w-sm rounded-xl bg-white p-5 text-gray-900 shadow-xl">
            <h2 className="mb-1 text-base font-semibold">添加项目</h2>
            <p className="mb-4 text-xs text-gray-500">输入项目目录的完整路径，该项目下的对话生成的文件都会写入此目录。</p>
            <label className="block mb-3">
              <span className="mb-1 block text-xs font-medium text-gray-500">项目目录</span>
              <input
                autoFocus
                placeholder="例如 C:\\Users\\me\\my-project"
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-mono"
                value={projectDialog.path}
                onChange={(e) => setProjectDialog((d) => (d ? { ...d, path: e.target.value } : d))}
                onKeyDown={(e) => { if (e.key === 'Enter') void submitProjectDialog(); if (e.key === 'Escape') setProjectDialog(null) }}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-gray-500">名称（可选）</span>
              <input
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                placeholder="默认使用文件夹名"
                value={projectDialog.name}
                onChange={(e) => setProjectDialog((d) => (d ? { ...d, name: e.target.value } : d))}
                onKeyDown={(e) => { if (e.key === 'Enter') void submitProjectDialog(); if (e.key === 'Escape') setProjectDialog(null) }}
              />
            </label>
            {projectsError && <div className="mt-3 text-xs text-red-500">{projectsError}</div>}
            <div className="mt-5 flex justify-end gap-2">
              <button onClick={() => setProjectDialog(null)} className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm">取消</button>
              <button onClick={() => void submitProjectDialog()} disabled={!projectDialog.path.trim()} className="rounded-lg bg-gray-900 px-4 py-1.5 text-sm text-white disabled:opacity-40">确定</button>
            </div>
          </div>
        </div>
      )}

      {/* Text Dialog */}
      {textDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4">
          <div className="w-full max-w-sm rounded-xl bg-white p-5 text-gray-900 shadow-xl">
            <h2 className="mb-4 text-base font-semibold">{textDialog.title}</h2>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-gray-500">{textDialog.label}</span>
              <input
                autoFocus
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                value={textValue}
                onChange={(e) => setTextValue(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void submitTextDialog(); if (e.key === 'Escape') closeTextDialog() }}
              />
            </label>
            <div className="mt-5 flex justify-end gap-2">
              <button onClick={closeTextDialog} className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm">取消</button>
              <button onClick={submitTextDialog} disabled={!textValue.trim()} className="rounded-lg bg-gray-900 px-4 py-1.5 text-sm text-white disabled:opacity-40">确定</button>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Dialog */}
      {confirmDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4">
          <div className="w-full max-w-sm rounded-xl bg-white p-5 text-gray-900 shadow-xl">
            <h2 className="mb-2 text-base font-semibold">{confirmDialog.title}</h2>
            <p className="text-sm text-gray-500">{confirmDialog.message}</p>
            <div className="mt-5 flex justify-end gap-2">
              <button onClick={() => setConfirmDialog(null)} className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm">取消</button>
              <button onClick={submitConfirmDialog} className="rounded-lg bg-red-600 px-4 py-1.5 text-sm text-white">确定</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/* ===== Memory View ===== */
function MemoryView() {
  const [memories, setMemories] = useState<{ id: string; theme: string; content: string; updated_at: string }[]>([])
  const [narrative, setNarrative] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    try {
      const res = await api.listMemories()
      setMemories(res.memories)
      setNarrative(res.narrative)
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])

  const handleDelete = async (id: string) => {
    await api.deleteMemory(id)
    void load()
  }

  const handleClear = async () => {
    await api.clearMemories()
    void load()
  }

  if (loading) return <div className="text-sm text-gray-400">加载中...</div>
  if (error) return <div className="text-sm text-red-500">{error}</div>

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold">长期记忆</h2>
        {memories.length > 0 && (
          <button onClick={handleClear} className="text-xs text-red-500 hover:text-red-700">清除全部</button>
        )}
      </div>
      {narrative && (
        <pre className="mb-4 p-3 rounded-lg bg-gray-50 border border-gray-200 text-xs text-gray-600 whitespace-pre-wrap">{narrative}</pre>
      )}
      {memories.length === 0 ? (
        <p className="text-sm text-gray-400">暂无长期记忆。对话中产生的持久事实会被异步整理到此处。</p>
      ) : (
        <div className="space-y-2">
          {memories.map((m) => (
            <div key={m.id} className="group flex items-start gap-3 rounded-lg border border-gray-200 bg-gray-50/50 p-3">
              <div className="flex-1 min-w-0">
                <span className="inline-block rounded bg-gray-200 px-1.5 py-0.5 text-[10px] font-medium text-gray-600 mb-1">{m.theme}</span>
                <p className="text-xs text-gray-700">{m.content}</p>
                <span className="text-[10px] text-gray-400 mt-1 block">{m.updated_at}</span>
              </div>
              <button onClick={() => void handleDelete(m.id)} className="p-1 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 shrink-0">
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ===== Models View ===== */
function ModelsView() {
  const settings = useSettingsStore((s) => s.settings)
  const openSettings = useSettingsStore((s) => s.setOpen)
  const providers = settings?.providers ?? []
  const activeId = settings?.active_provider_id ?? 'default'

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold">模型提供商</h2>
        <button onClick={() => openSettings(true)} className="text-xs text-blue-600 hover:text-blue-800">管理设置</button>
      </div>
      <div className="space-y-2">
        {providers.map((p) => {
          const isActive = p.id === activeId
          return (
            <div
              key={p.id}
              className={`rounded-lg border p-3 ${isActive ? 'border-blue-300 bg-blue-50/40' : 'border-gray-200 bg-gray-50/50'}`}
            >
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-blue-500' : 'bg-gray-300'}`} />
                <span className="text-sm font-medium flex-1">{p.label}</span>
                <span className="text-xs text-gray-500 font-mono">{p.model}</span>
                <span className={`text-xs ${p.has_api_key ? 'text-green-500' : 'text-red-400'}`}>
                  {p.has_api_key ? '🔑' : '🔓'}
                </span>
                {isActive && <span className="text-[10px] bg-blue-100 text-blue-600 px-1.5 py-0.5 rounded">使用中</span>}
              </div>
              <div className="mt-1 text-[10px] text-gray-400 font-mono">{p.base_url}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}