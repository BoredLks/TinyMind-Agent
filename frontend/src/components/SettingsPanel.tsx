import { useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, Plus, RefreshCw, TestTube, Trash2 } from 'lucide-react'

import { api } from '../api/restClient'
import { type AppSettings, useSettingsStore } from '../stores/settingsStore'

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="mb-3 block">
      <span className="mb-1 block text-xs font-medium text-gray-500 dark:text-gray-400">{label}</span>
      {children}
    </label>
  )
}

const inputCls =
  'w-full rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm dark:border-gray-600 dark:bg-gray-700'
const smallInputCls =
  'w-full rounded border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-600 dark:bg-gray-700'

export default function SettingsPanel() {
  const open = useSettingsStore((s) => s.open)
  const setOpen = useSettingsStore((s) => s.setOpen)
  const settings = useSettingsStore((s) => s.settings)
  const save = useSettingsStore((s) => s.save)
  const loadSettings = useSettingsStore((s) => s.load)

  const [form, setForm] = useState<AppSettings | null>(null)
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showAddProvider, setShowAddProvider] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [providerModels, setProviderModels] = useState<Record<string, string[]>>({})
  const [addForm, setAddForm] = useState({ id: '', label: '', base_url: '', model: '', api_key: '' })
  const [editKey, setEditKey] = useState<Record<string, string>>({})

  useEffect(() => {
    if (settings) setForm({ ...settings })
  }, [settings, open])

  if (!open || !form) return null

  const upd = <K extends keyof AppSettings>(k: K, v: AppSettings[K]) =>
    setForm((f) => (f ? { ...f, [k]: v } : f))

  const numOrNull = (v: string) => (v.trim() === '' ? null : Number(v))
  const msg = (err: unknown) => (err instanceof Error ? err.message : String(err))

  const markKey = (id: string, has: boolean) => {
    setForm((f) =>
      f
        ? {
            ...f,
            has_api_key: id === 'default' ? has : f.has_api_key,
            providers: f.providers.map((p) => (p.id === id ? { ...p, has_api_key: has } : p)),
          }
        : f,
    )
  }

  const doTest = async (id: string) => {
    const p = form.providers.find((x) => x.id === id)
    if (!p) return
    setBusy(`test:${id}`)
    try {
      const key = editKey[id]?.trim()
      const res = await api.testProvider({ id, base_url: p.base_url, model: p.model, api_key: key || undefined })
      setNotice(`${p.label}: ${res.ok ? '✅ 可用' : '❌ 失败'} · ${res.latency_ms}ms · ${res.detail}`)
    } catch (err) {
      setNotice(`${p.label}: 测试失败 · ${msg(err)}`)
    } finally {
      setBusy('')
    }
  }

  const doDiscover = async (id: string) => {
    const p = form.providers.find((x) => x.id === id)
    if (!p) return
    setBusy(`discover:${id}`)
    try {
      const key = editKey[id]?.trim()
      const res = await api.discoverProviderModels({ id, base_url: p.base_url, api_key: key || undefined })
      if (!res.ok) {
        setNotice(`${p.label}: 拉取失败 · ${res.detail}`)
        return
      }
      setProviderModels((cur) => ({ ...cur, [id]: res.models }))
      setNotice(`${p.label}: 发现 ${res.models.length} 个模型 · ${res.latency_ms}ms`)
    } catch (err) {
      setNotice(`${p.label}: 拉取失败 · ${msg(err)}`)
    } finally {
      setBusy('')
    }
  }

  const doSwitchModel = async (id: string, model: string) => {
    setBusy(`model:${id}`)
    try {
      await api.updateProvider(id, { model })
      await loadSettings()
      setProviderNotice(`已切换到 ${model}`)
    } catch (err) {
      setNotice(`切换失败 · ${msg(err)}`)
    } finally {
      setBusy('')
    }
  }

  const setProviderNotice = (s: string) => setNotice(s)

  const doSaveKey = async (id: string) => {
    const key = (editKey[id] || '').trim()
    if (!key) return
    setBusy(`key:${id}`)
    try {
      if (id === 'default') {
        await api.setApiKey(key)
      } else {
        await api.updateProvider(id, { api_key: key })
      }
      setEditKey((f) => ({ ...f, [id]: '' }))
      markKey(id, true)
      setNotice(`${id === 'default' ? '默认' : id}: API Key 已保存`)
    } catch (err) {
      setNotice(`保存失败 · ${msg(err)}`)
    } finally {
      setBusy('')
    }
  }

  const doDelete = async (id: string) => {
    try {
      await api.deleteProvider(id)
      await loadSettings()
      setNotice('已删除')
    } catch (err) {
      setNotice(`删除失败 · ${msg(err)}`)
    }
  }

  const doActivate = async (id: string) => {
    setBusy(`activate:${id}`)
    try {
      if (id === 'default') {
        upd('active_provider_id', 'default')
      } else {
        await api.activateProvider(id)
        await loadSettings()
      }
      const p = form.providers.find((x) => x.id === id)
      setNotice(`已切换到 ${p?.label ?? id}，将在下一次对话生效`)
    } catch (err) {
      setNotice(`切换失败 · ${msg(err)}`)
    } finally {
      setBusy('')
    }
  }

  const doAddProvider = async () => {
    setBusy('add')
    try {
      await api.createProvider({
        id: addForm.id.trim(),
        label: addForm.label.trim() || addForm.id.trim(),
        base_url: addForm.base_url.trim(),
        model: addForm.model.trim() || providerModels.__add?.[0] || 'default',
        api_key: addForm.api_key.trim() || undefined,
      })
      setAddForm({ id: '', label: '', base_url: '', model: '', api_key: '' })
      setShowAddProvider(false)
      setProviderModels({})
      await loadSettings()
      setNotice('提供商已添加')
    } catch (err) {
      setNotice(`添加失败 · ${msg(err)}`)
    } finally {
      setBusy('')
    }
  }

  const onSave = async () => {
    const patch: Partial<AppSettings> = {
      temperature: Number(form.temperature),
      top_p: form.top_p,
      max_tokens: form.max_tokens,
      active_provider_id: form.active_provider_id,
      system_prompt: form.system_prompt || null,
      context_max_messages: Number(form.context_max_messages),
      context_strategy: form.context_strategy,
      context_max_tokens: Number(form.context_max_tokens) || 0,
      max_tool_iterations: Number(form.max_tool_iterations) || 30,
      max_tool_arg_len: Number(form.max_tool_arg_len) || 4000,
      max_tool_result_len: Number(form.max_tool_result_len) || 6000,
      max_tool_text_len: Number(form.max_tool_text_len) || 8000,
      api_timeout: Number(form.api_timeout) || 300,
      turn_timeout: Number(form.turn_timeout) || 600,
      theme: form.theme,
      workspace_root: form.workspace_root,
      memory_enabled: form.memory_enabled,
    }
    if (form.active_provider_id === 'default') {
      patch.model = form.model
      patch.base_url = form.base_url
    }
    setBusy('save')
    try {
      await save(patch)
      // Save any pending keys
      for (const id of Object.keys(editKey)) {
        if (editKey[id]?.trim()) {
          await doSaveKey(id)
        }
      }
      await loadSettings()
      setNotice('设置已保存')
      setOpen(false)
    } catch (err) {
      setNotice(`保存失败 · ${msg(err)}`)
    } finally {
      setBusy('')
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
      onClick={() => setOpen(false)}
    >
      <div
        className="max-h-[85vh] w-[600px] overflow-y-auto rounded-2xl bg-white p-6 text-gray-900 shadow-xl dark:bg-gray-800 dark:text-gray-100"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="mb-4 text-lg font-semibold">设置</h2>

        {/* === LLM Providers Section === */}
        <div className="mb-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="text-sm font-medium">LLM 提供商</span>
            <button
              onClick={() => setShowAddProvider(!showAddProvider)}
              className="inline-flex items-center gap-1 rounded-lg border border-gray-300 px-2 py-1 text-xs hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-700"
            >
              <Plus size={12} />
              添加提供商
            </button>
          </div>

          {/* Provider Cards */}
          <div className="space-y-2">
            {form.providers.map((p) => {
              const isActive = form.active_provider_id === p.id
              const isEditing = editingId === p.id
              return (
                <div
                  key={p.id}
                  className={`rounded-xl border p-3 transition-colors ${
                    isActive
                      ? 'border-blue-400 bg-blue-50/50 dark:border-blue-600 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  {/* Provider Header */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => void doActivate(p.id)}
                      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2 ${
                        isActive
                          ? 'border-blue-500 bg-blue-500'
                          : 'border-gray-300 dark:border-gray-600'
                      }`}
                      title={isActive ? '当前使用中' : '切换到此提供商'}
                    >
                      {isActive && <div className="h-1.5 w-1.5 rounded-full bg-white" />}
                    </button>
                    <span className="min-w-0 flex-1 truncate text-sm font-medium">
                      {p.label}
                      {isActive && <span className="ml-2 text-xs text-blue-500">使用中</span>}
                    </span>
                    <span className="text-xs text-gray-400">{p.model}</span>
                    <span className={`text-xs ${p.has_api_key ? 'text-green-500' : 'text-red-400'}`}>
                      {p.has_api_key ? '🔑' : '🔓'}
                    </span>
                    <button
                      onClick={() => void doTest(p.id)}
                      disabled={busy === `test:${p.id}`}
                      className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-700"
                      title="测速"
                    >
                      <TestTube size={14} />
                    </button>
                    <button
                      onClick={() => setEditingId(isEditing ? null : p.id)}
                      className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-700"
                      title="编辑"
                    >
                      {isEditing ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    </button>
                    {p.id !== 'default' && (
                      <button
                        onClick={() => void doDelete(p.id)}
                        className="rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-500 dark:hover:bg-red-900/20"
                        title="删除"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>

                  {/* Provider Edit Form (expanded) */}
                  {isEditing && (
                    <div className="mt-3 space-y-2 border-t border-gray-200 pt-3 dark:border-gray-700">
                      {p.id === 'default' ? (
                        <>
                          <div className="grid grid-cols-2 gap-2">
                            <input
                              className={smallInputCls}
                              placeholder="base_url"
                              value={form.base_url}
                              onChange={(e) => upd('base_url', e.target.value)}
                            />
                            <input
                              className={smallInputCls}
                              placeholder="model"
                              value={form.model}
                              onChange={(e) => upd('model', e.target.value)}
                            />
                          </div>
                        </>
                      ) : (
                        <div className="text-xs text-gray-500">
                          {p.label} · {p.base_url} · {p.model}
                        </div>
                      )}

                      {/* API Key */}
                      <div className="flex items-center gap-2">
                        <input
                          className={smallInputCls + ' flex-1'}
                          type="password"
                          placeholder={p.has_api_key ? '已设置 Key（留空不修改）' : '输入 API Key'}
                          value={editKey[p.id] ?? ''}
                          onChange={(e) => setEditKey((f) => ({ ...f, [p.id]: e.target.value }))}
                        />
                        <button
                          onClick={() => void doSaveKey(p.id)}
                          disabled={busy === `key:${p.id}`}
                          className="shrink-0 rounded bg-gray-900 px-2 py-1 text-xs text-white dark:bg-gray-100 dark:text-gray-900"
                        >
                          保存 Key
                        </button>
                      </div>

                      {/* Discover + Model Select */}
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => void doDiscover(p.id)}
                          disabled={busy === `discover:${p.id}`}
                          className="inline-flex items-center gap-1 rounded border border-gray-300 px-2 py-1 text-xs dark:border-gray-600"
                        >
                          <RefreshCw size={12} className={busy === `discover:${p.id}` ? 'animate-spin' : ''} />
                          拉取模型
                        </button>
                        {providerModels[p.id]?.length ? (
                          <select
                            className={smallInputCls + ' max-w-[200px]'}
                            value={p.model}
                            onChange={(e) => void doSwitchModel(p.id, e.target.value)}
                          >
                            {providerModels[p.id].map((m) => (
                              <option key={m} value={m}>{m}</option>
                            ))}
                          </select>
                        ) : null}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Add Provider Form */}
          {showAddProvider && (
            <div className="mt-3 rounded-xl border border-dashed border-gray-300 p-3 dark:border-gray-600">
              <div className="mb-2 text-xs font-medium text-gray-500">添加新提供商</div>
              <div className="grid grid-cols-2 gap-2">
                <input
                  className={smallInputCls}
                  placeholder="ID（例如 openai）"
                  value={addForm.id}
                  onChange={(e) => setAddForm((f) => ({ ...f, id: e.target.value }))}
                />
                <input
                  className={smallInputCls}
                  placeholder="显示名称"
                  value={addForm.label}
                  onChange={(e) => setAddForm((f) => ({ ...f, label: e.target.value }))}
                />
                <input
                  className={smallInputCls}
                  placeholder="base_url"
                  value={addForm.base_url}
                  onChange={(e) => setAddForm((f) => ({ ...f, base_url: e.target.value }))}
                />
                <input
                  className={smallInputCls}
                  placeholder="model"
                  value={addForm.model}
                  onChange={(e) => setAddForm((f) => ({ ...f, model: e.target.value }))}
                />
                <input
                  className={smallInputCls + ' col-span-2'}
                  type="password"
                  placeholder="API Key"
                  value={addForm.api_key}
                  onChange={(e) => setAddForm((f) => ({ ...f, api_key: e.target.value }))}
                />
              </div>
              <div className="mt-2 flex items-center justify-between">
                <div className="flex gap-2">
                  <button
                    className="rounded border border-gray-300 px-2 py-1 text-xs dark:border-gray-600"
                    disabled={!addForm.base_url.trim() || !addForm.api_key.trim()}
                    onClick={async () => {
                      setBusy('discover:add')
                      try {
                        const res = await api.discoverProviderModels({
                          base_url: addForm.base_url.trim(),
                          api_key: addForm.api_key.trim(),
                        })
                        if (res.ok) {
                          setProviderModels((cur) => ({ ...cur, __add: res.models }))
                          setAddForm((f) => ({ ...f, model: f.model || res.models[0] || '' }))
                          setNotice(`发现 ${res.models.length} 个模型`)
                        } else {
                          setNotice(`拉取失败 · ${res.detail}`)
                        }
                      } catch (err) {
                        setNotice(`拉取失败 · ${msg(err)}`)
                      } finally {
                        setBusy('')
                      }
                    }}
                  >
                    {busy === 'discover:add' ? '拉取中...' : '拉取模型'}
                  </button>
                  <button
                    className="rounded bg-gray-900 px-3 py-1 text-xs text-white dark:bg-gray-100 dark:text-gray-900"
                    disabled={!addForm.id.trim() || !addForm.base_url.trim()}
                    onClick={() => void doAddProvider()}
                  >
                    {busy === 'add' ? '添加中...' : '确认添加'}
                  </button>
                  <button
                    className="rounded border border-gray-300 px-2 py-1 text-xs dark:border-gray-600"
                    onClick={() => { setShowAddProvider(false); setAddForm({ id: '', label: '', base_url: '', model: '', api_key: '' }); setProviderModels({}) }}
                  >
                    取消
                  </button>
                </div>
                {providerModels.__add?.length ? (
                  <select
                    className={smallInputCls + ' max-w-[180px]'}
                    value={addForm.model}
                    onChange={(e) => setAddForm((f) => ({ ...f, model: e.target.value }))}
                  >
                    {providerModels.__add.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                ) : null}
              </div>
            </div>
          )}
        </div>

        {/* === General Settings (collapsible) === */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="mb-3 flex items-center gap-1 text-xs font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
        >
          {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          通用设置
        </button>

        {showAdvanced && (
          <div className="mb-4 space-y-3 rounded-lg border border-gray-200 p-3 dark:border-gray-700">
            <div className="grid grid-cols-3 gap-3">
              <Field label="温度">
                <input
                  className={inputCls}
                  type="number"
                  step="0.1"
                  value={form.temperature}
                  onChange={(e) => upd('temperature', Number(e.target.value))}
                />
              </Field>
              <Field label="top_p">
                <input
                  className={inputCls}
                  type="number"
                  step="0.05"
                  value={form.top_p ?? ''}
                  onChange={(e) => upd('top_p', numOrNull(e.target.value))}
                />
              </Field>
              <Field label="max_tokens">
                <input
                  className={inputCls}
                  type="number"
                  value={form.max_tokens ?? ''}
                  onChange={(e) => upd('max_tokens', numOrNull(e.target.value))}
                />
              </Field>
            </div>

            <Field label="系统提示词">
              <textarea
                className={inputCls}
                rows={3}
                value={form.system_prompt ?? ''}
                onChange={(e) => upd('system_prompt', e.target.value)}
              />
            </Field>

            <Field label="当前项目 / 工具工作区目录">
              <input
                className={inputCls}
                value={form.workspace_root}
                onChange={(e) => upd('workspace_root', e.target.value)}
              />
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="最大工具调用轮次">
                <input
                  className={inputCls}
                  type="number"
                  step="5"
                  placeholder="默认 30"
                  value={form.max_tool_iterations}
                  onChange={(e) => upd('max_tool_iterations', Number(e.target.value) || 30)}
                />
              </Field>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <Field label="工具参数截断 (字符)">
                <input
                  className={inputCls}
                  type="number"
                  step="1000"
                  placeholder="4000"
                  value={form.max_tool_arg_len}
                  onChange={(e) => upd('max_tool_arg_len', Number(e.target.value) || 4000)}
                />
              </Field>
              <Field label="工具结果截断 (字符)">
                <input
                  className={inputCls}
                  type="number"
                  step="1000"
                  placeholder="6000"
                  value={form.max_tool_result_len}
                  onChange={(e) => upd('max_tool_result_len', Number(e.target.value) || 6000)}
                />
              </Field>
              <Field label="回复截断 (字符)">
                <input
                  className={inputCls}
                  type="number"
                  step="1000"
                  placeholder="8000"
                  value={form.max_tool_text_len}
                  onChange={(e) => upd('max_tool_text_len', Number(e.target.value) || 8000)}
                />
              </Field>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Field label="上下文策略">
                <select
                  className={inputCls}
                  value={form.context_strategy}
                  onChange={(e) => upd('context_strategy', e.target.value)}
                >
                  <option value="sliding_window">滑动窗口（按消息数）</option>
                  <option value="token_limit">Token 限制（按 token 数）</option>
                </select>
              </Field>
              {form.context_strategy === 'token_limit' ? (
                <Field label="上下文最大 Token 数">
                  <input
                    className={inputCls}
                    type="number"
                    step="1000"
                    placeholder="例如 1000000"
                    value={form.context_max_tokens || ''}
                    onChange={(e) => upd('context_max_tokens', Number(e.target.value) || 0)}
                  />
                </Field>
              ) : (
                <Field label="上下文最大消息数">
                  <input
                    className={inputCls}
                    type="number"
                    value={form.context_max_messages}
                    onChange={(e) => upd('context_max_messages', Number(e.target.value))}
                  />
                </Field>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Field label="API 超时 (秒)">
                <input
                  className={inputCls}
                  type="number"
                  step="60"
                  placeholder="默认 300"
                  value={form.api_timeout}
                  onChange={(e) => upd('api_timeout', Number(e.target.value) || 300)}
                />
              </Field>
              <Field label="轮次超时 (秒)">
                <input
                  className={inputCls}
                  type="number"
                  step="60"
                  placeholder="默认 900"
                  value={form.turn_timeout}
                  onChange={(e) => upd('turn_timeout', Number(e.target.value) || 900)}
                />
              </Field>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Field label="主题">
                <select
                  aria-label="主题"
                  className={inputCls}
                  value={form.theme}
                  onChange={(e) => upd('theme', e.target.value as 'light' | 'dark')}
                >
                  <option value="light">浅色</option>
                  <option value="dark">深色</option>
                </select>
              </Field>
            </div>

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.memory_enabled}
                onChange={(e) => upd('memory_enabled', e.target.checked)}
              />
              <span>启用后台长期记忆整理</span>
            </label>
          </div>
        )}

        {/* Notice bar */}
        {notice && (
          <div className="mb-3 rounded-lg bg-gray-100 px-3 py-2 text-xs text-gray-600 dark:bg-gray-700 dark:text-gray-300">
            {notice}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end gap-2">
          <button
            onClick={() => setOpen(false)}
            className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm dark:border-gray-600"
          >
            取消
          </button>
          <button
            onClick={onSave}
            disabled={busy === 'save'}
            className="rounded-lg bg-gray-900 px-4 py-1.5 text-sm text-white dark:bg-gray-100 dark:text-gray-900"
          >
            {busy === 'save' ? '保存中...' : '保存'}
          </button>
        </div>
      </div>
    </div>
  )
}