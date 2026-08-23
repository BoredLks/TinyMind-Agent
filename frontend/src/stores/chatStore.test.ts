import { beforeEach, describe, expect, it } from 'vitest'

import { useChatStore } from './chatStore'

beforeEach(() => {
  useChatStore.getState().reset()
  // reset() intentionally preserves background-session state (generating /
  // _liveItems) in production; wipe it here for per-test isolation.
  useChatStore.setState({ viewedId: '', generating: {}, _liveItems: {} })
})

describe('chatStore timeline', () => {
  it('streams assistant text into a single message item', () => {
    const s = useChatStore.getState()
    s.pushUser('hi')
    s.appendAssistantText('Hel')
    s.appendAssistantText('lo')

    const { items } = useChatStore.getState()
    expect(items).toHaveLength(2)
    expect(items[0]).toMatchObject({ kind: 'message', role: 'user', content: 'hi' })
    expect(items[1]).toMatchObject({ kind: 'message', role: 'assistant', content: 'Hello' })
  })

  it('interleaves a tool card between assistant text', () => {
    const s = useChatStore.getState()
    s.pushUser('do it')
    s.appendAssistantText('thinking')
    s.addToolCall('c1', 'write_file', '{}')
    s.setToolAwaitingApproval('c1')
    s.setToolResult('c1', true, 'wrote 3 chars')
    s.appendAssistantText('done')

    const { items } = useChatStore.getState()
    expect(items.map((i) => i.kind)).toEqual(['message', 'message', 'tool', 'message'])
    expect(items[2]).toMatchObject({
      kind: 'tool',
      callId: 'c1',
      name: 'write_file',
      state: 'ok',
      result: 'wrote 3 chars',
    })
    expect(items[3]).toMatchObject({ role: 'assistant', content: 'done' })
  })

  it('setHistory replaces items with message bubbles', () => {
    useChatStore.getState().setHistory([
      { id: 'a', role: 'user', content: 'hi' },
      { id: 'b', role: 'assistant', content: 'yo' },
    ])
    const { items, status } = useChatStore.getState()
    expect(items.map((i) => (i.kind === 'message' ? i.content : ''))).toEqual(['hi', 'yo'])
    expect(status).toBe('idle')
  })

  it('records errors and flips status', () => {
    useChatStore.getState().setError('boom')
    expect(useChatStore.getState().status).toBe('error')
    expect(useChatStore.getState().error).toBe('boom')
  })

  it('keeps a background session stream out of the viewed timeline', () => {
    const s = useChatStore.getState()
    s.setViewedId('A')
    s.pushUser('hi A', 'A')
    s.appendAssistantText('answer A', 'A')
    // A different session streams while NOT viewed — must not touch `items`.
    s.appendReasoning('thinking B', 'B')
    s.appendAssistantText('answer B', 'B')

    const viewed = useChatStore.getState().items
    expect(viewed).toHaveLength(2)
    expect(viewed[1]).toMatchObject({ role: 'assistant', content: 'answer A' })
    // No 'answer B' / 'thinking B' leaked into the viewed session.
    expect(JSON.stringify(viewed)).not.toContain('answer B')
    expect(JSON.stringify(viewed)).not.toContain('thinking B')
  })

  it('restores a background session timeline when it becomes viewed', () => {
    const s = useChatStore.getState()
    s.setViewedId('A')
    s.appendAssistantText('answer B', 'B') // buffered for B while viewing A
    // Switch to B: snapshot A, then restore B's buffer.
    s.saveLiveItems('A')
    s.setViewedId('B')
    expect(s.restoreLiveItems('B')).toBe(true)

    const items = useChatStore.getState().items
    expect(items).toHaveLength(1)
    expect(items[0]).toMatchObject({ role: 'assistant', content: 'answer B' })
  })

  it('tracks generating state per session and drives viewed status', () => {
    const s = useChatStore.getState()
    s.setViewedId('A')
    s.setGenerating('B', true) // background session busy — viewed status unaffected
    expect(useChatStore.getState().status).toBe('idle')
    s.setGenerating('A', true) // viewed session busy — status reflects it
    expect(useChatStore.getState().status).toBe('generating')
    expect(s.isGenerating('B')).toBe(true)
    s.setGenerating('A', false)
    expect(useChatStore.getState().status).toBe('idle')
  })
})
