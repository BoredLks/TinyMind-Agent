import { useState } from 'react'
import { CornerDownRight, Folder, Pencil, Plus, Trash2 } from 'lucide-react'

import { useProjectsStore } from '../stores/projectsStore'
import { useSessionsStore } from '../stores/sessionsStore'
import { useSettingsStore } from '../stores/settingsStore'

function parseSessionMeta(metaJson?: string | null): Record<string, unknown> {
  if (!metaJson) return {}
  try {
    const parsed = JSON.parse(metaJson)
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

export default function Sidebar() {
  const projects = useProjectsStore((s) => s.projects)
  const currentProjectId = useProjectsStore((s) => s.currentId)
  const selectProject = useProjectsStore((s) => s.select)
  const addProject = useProjectsStore((s) => s.add)
  const renameProject = useProjectsStore((s) => s.rename)
  const removeProject = useProjectsStore((s) => s.remove)
  const projectError = useProjectsStore((s) => s.error)
  const clearProjectError = useProjectsStore((s) => s.clearError)
  const workspaceRoot = useSettingsStore((s) => s.settings?.workspace_root ?? '')
  const sessions = useSessionsStore((s) => s.sessions)
  const currentId = useSessionsStore((s) => s.currentId)
  const select = useSessionsStore((s) => s.select)
  const create = useSessionsStore((s) => s.create)
  const rename = useSessionsStore((s) => s.rename)
  const remove = useSessionsStore((s) => s.remove)
  const [textDialog, setTextDialog] = useState<
    | { kind: 'addProject'; title: string; label: string; initial: string }
    | { kind: 'renameProject'; title: string; label: string; initial: string; id: string }
    | { kind: 'renameSession'; title: string; label: string; initial: string; id: string }
    | null
  >(null)
  const [textValue, setTextValue] = useState('')
  const [confirmDialog, setConfirmDialog] = useState<
    | { kind: 'deleteProject'; title: string; message: string; id: string }
    | { kind: 'deleteSession'; title: string; message: string; id: string }
    | null
  >(null)

  const openTextDialog = (dialog: NonNullable<typeof textDialog>) => {
    setTextDialog(dialog)
    setTextValue(dialog.initial)
  }

  const closeTextDialog = () => {
    setTextDialog(null)
    setTextValue('')
  }

  const submitTextDialog = async () => {
    const value = textValue.trim()
    if (!textDialog || !value) return
    if (textDialog.kind === 'addProject') await addProject(value)
    if (textDialog.kind === 'renameProject') await renameProject(textDialog.id, value)
    if (textDialog.kind === 'renameSession') await rename(textDialog.id, value)
    closeTextDialog()
  }

  const submitConfirmDialog = async () => {
    if (!confirmDialog) return
    if (confirmDialog.kind === 'deleteProject') await removeProject(confirmDialog.id)
    if (confirmDialog.kind === 'deleteSession') await remove(confirmDialog.id)
    setConfirmDialog(null)
  }

  return (
    <aside className="flex w-64 flex-col border-r border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800">
      <div className="p-3">
        <button
          data-testid="add-project-button"
          onClick={() =>
            openTextDialog({
              kind: 'addProject',
              title: '添加项目',
              label: '项目目录完整路径',
              initial: workspaceRoot,
            })
          }
          className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-gray-900 px-3 py-2 text-sm font-medium text-white dark:bg-gray-100 dark:text-gray-900"
        >
          <Plus size={15} />
          添加项目
        </button>
      </div>

      <div className="border-b border-gray-200 px-2 pb-3 dark:border-gray-700">
        <div className="mb-2 px-2 text-xs font-medium text-gray-400">项目</div>
        {projectError && (
          <button
            onClick={clearProjectError}
            className="mb-2 w-full rounded-md bg-red-50 px-2 py-1 text-left text-xs text-red-600 dark:bg-red-900/30 dark:text-red-200"
            title="点击关闭"
          >
            {projectError}
          </button>
        )}
        <div className="max-h-44 overflow-y-auto">
          {projects.length === 0 && (
            <div className="px-2 py-2 text-xs text-gray-400">
              还没有项目。添加后，AI 只能在选中的项目目录里生成文件。
            </div>
          )}
          {projects.map((p) => {
            const active = p.id === currentProjectId
            return (
              <div
                key={p.id}
                onClick={() => selectProject(p.id)}
                title={p.path}
                className={
                  'group flex cursor-pointer items-center justify-between rounded-lg px-2 py-2 text-sm ' +
                  (active
                    ? 'bg-gray-200 dark:bg-gray-700'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700/50')
                }
              >
                <span className="flex min-w-0 flex-1 items-center gap-2">
                  <Folder size={15} className="shrink-0 text-gray-500" />
                  <span className="truncate">{p.name}</span>
                </span>
                <span className="ml-2 flex gap-2 opacity-0 transition group-hover:opacity-100 group-focus-within:opacity-100">
                  <button
                    aria-label={`重命名项目 ${p.name}`}
                    title="重命名项目"
                    onClick={(e) => {
                      e.stopPropagation()
                      openTextDialog({
                        kind: 'renameProject',
                        title: '重命名项目',
                        label: '项目名称',
                        initial: p.name,
                        id: p.id,
                      })
                    }}
                    className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                  >
                    <Pencil size={14} />
                  </button>
                  <button
                    aria-label={`移除项目 ${p.name}`}
                    title="移除项目"
                    onClick={(e) => {
                      e.stopPropagation()
                      setConfirmDialog({
                        kind: 'deleteProject',
                        title: '移除项目',
                        message: '只会从项目列表移除，不会删除磁盘文件。',
                        id: p.id,
                      })
                    }}
                    className="text-gray-400 hover:text-red-500"
                  >
                    <Trash2 size={14} />
                  </button>
                </span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="border-b border-gray-200 p-3 dark:border-gray-700">
        <button
          onClick={() => create()}
          className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium dark:border-gray-600"
        >
          <Plus size={15} />
          新会话
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-3">
        <div className="mb-2 px-2 pt-2 text-xs font-medium text-gray-400">会话</div>
        {sessions.map((s) => {
          const active = s.id === currentId
          const meta = parseSessionMeta(s.meta_json)
          const isSubagent = meta.is_subagent === true
          return (
            <div
              key={s.id}
              onClick={() => select(s.id)}
              className={
                'group flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm ' +
                (active
                  ? 'bg-gray-200 dark:bg-gray-700'
                : 'hover:bg-gray-100 dark:hover:bg-gray-700/50')
              }
            >
              <span className="flex min-w-0 flex-1 items-center gap-2">
                {isSubagent && (
                  <CornerDownRight
                    size={14}
                    className="shrink-0 text-gray-400"
                    aria-hidden="true"
                  />
                )}
                <span className="truncate">{s.title}</span>
              </span>
              <span className="ml-2 flex gap-2 opacity-0 transition group-hover:opacity-100 group-focus-within:opacity-100">
                <button
                  aria-label={`重命名会话 ${s.title}`}
                  title="重命名"
                  onClick={(e) => {
                    e.stopPropagation()
                    openTextDialog({
                      kind: 'renameSession',
                      title: '重命名会话',
                      label: '会话名称',
                      initial: s.title,
                      id: s.id,
                    })
                  }}
                  className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                >
                  <Pencil size={14} />
                </button>
                <button
                  aria-label={`删除会话 ${s.title}`}
                  title="删除"
                  onClick={(e) => {
                    e.stopPropagation()
                    setConfirmDialog({
                      kind: 'deleteSession',
                      title: '删除会话',
                      message: '会删除该会话及其全部消息。',
                      id: s.id,
                    })
                  }}
                  className="text-gray-400 hover:text-red-500"
                >
                  <Trash2 size={14} />
                </button>
              </span>
            </div>
          )
        })}
      </div>
      {textDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4">
          <div className="w-full max-w-md rounded-xl bg-white p-5 text-gray-900 shadow-xl dark:bg-gray-800 dark:text-gray-100">
            <h2 className="mb-4 text-base font-semibold">{textDialog.title}</h2>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-gray-500 dark:text-gray-400">
                {textDialog.label}
              </span>
              <input
                autoFocus
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700"
                value={textValue}
                onChange={(e) => setTextValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') void submitTextDialog()
                  if (e.key === 'Escape') closeTextDialog()
                }}
              />
            </label>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={closeTextDialog}
                className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm dark:border-gray-600"
              >
                取消
              </button>
              <button
                onClick={submitTextDialog}
                disabled={!textValue.trim()}
                className="rounded-lg bg-gray-900 px-4 py-1.5 text-sm text-white disabled:opacity-40 dark:bg-gray-100 dark:text-gray-900"
              >
                确定
              </button>
            </div>
          </div>
        </div>
      )}
      {confirmDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4">
          <div className="w-full max-w-sm rounded-xl bg-white p-5 text-gray-900 shadow-xl dark:bg-gray-800 dark:text-gray-100">
            <h2 className="mb-2 text-base font-semibold">{confirmDialog.title}</h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">{confirmDialog.message}</p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setConfirmDialog(null)}
                className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm dark:border-gray-600"
              >
                取消
              </button>
              <button
                onClick={submitConfirmDialog}
                className="rounded-lg bg-red-600 px-4 py-1.5 text-sm text-white"
              >
                确定
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}
