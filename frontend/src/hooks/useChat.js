import { useState, useCallback, useRef } from 'react'
import { api } from '../api'

const newConversation = (id) => ({
  id,
  title: 'Cuộc trò chuyện mới',
  messages: [],
  createdAt: Date.now(),
})

export function useChat() {
  const [conversations, setConversations] = useState([newConversation(Date.now())])
  const [activeId, setActiveId] = useState(conversations[0].id)
  const [loading, setLoading] = useState(false)
  const [settings, setSettings] = useState({
    useRag: true,
    useRerank: false,
    useHybrid: false,
    useReflection: false,
  })
  const abortRef = useRef(null)

  const active = conversations.find(c => c.id === activeId) || conversations[0]

  const updateActive = useCallback((fn) => {
    setConversations(prev => prev.map(c => c.id === activeId ? fn(c) : c))
  }, [activeId])

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return

    const userMsg = { role: 'user', content: text, id: Date.now() }
    updateActive(c => ({
      ...c,
      title: c.messages.length === 0 ? text.slice(0, 40) : c.title,
      messages: [...c.messages, userMsg],
    }))

    setLoading(true)
    const history = [...active.messages, userMsg].map(m => ({
      role: m.role,
      content: m.content,
    }))

    try {
      const res = await api.chat({
        messages: history,
        use_rag: settings.useRag,
        use_rerank: settings.useRerank,
        use_hybrid: settings.useHybrid,
        use_reflection: settings.useReflection,
      })

      const assistantMsg = {
        role: 'assistant',
        content: res.answer,
        sources: res.sources || [],
        id: Date.now() + 1,
      }
      updateActive(c => ({ ...c, messages: [...c.messages, assistantMsg] }))
    } catch (err) {
      const errMsg = {
        role: 'assistant',
        content: `Lỗi: ${err.response?.data?.detail || err.message}`,
        error: true,
        id: Date.now() + 1,
      }
      updateActive(c => ({ ...c, messages: [...c.messages, errMsg] }))
    } finally {
      setLoading(false)
    }
  }, [loading, active, settings, updateActive])

  const newChat = useCallback(() => {
    const c = newConversation(Date.now())
    setConversations(prev => [c, ...prev])
    setActiveId(c.id)
  }, [])

  const deleteConversation = useCallback((id) => {
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id)
      if (next.length === 0) {
        const fresh = newConversation(Date.now())
        setActiveId(fresh.id)
        return [fresh]
      }
      if (id === activeId) setActiveId(next[0].id)
      return next
    })
  }, [activeId])

  const clearMessages = useCallback(() => {
    updateActive(c => ({ ...c, messages: [], title: 'Cuộc trò chuyện mới' }))
  }, [updateActive])

  return {
    conversations,
    active,
    activeId,
    setActiveId,
    loading,
    settings,
    setSettings,
    sendMessage,
    newChat,
    deleteConversation,
    clearMessages,
  }
}
