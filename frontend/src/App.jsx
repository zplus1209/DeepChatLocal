import { useEffect, useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import SettingsDrawer from './components/SettingsDrawer'
import { useChat } from './hooks/useChat'
import { api } from './api'

export default function App() {
  const [dbStatus, setDbStatus] = useState(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [toast, setToast] = useState(null)

  const {
    conversations, active, activeId,
    setActiveId, loading, settings, setSettings,
    sendMessage, newChat, deleteConversation, clearMessages
  } = useChat()

  // Health check
  useEffect(() => {
    api.health()
      .then(setDbStatus)
      .catch(() => setDbStatus(null))
  }, [])

  const showToast = useCallback((msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }, [])

  const handleUpload = useCallback(async (file, onProgress) => {
    try {
      const res = await api.ingestFile(file, onProgress)
      showToast(`Đã nhập ${res.count} đoạn từ "${file.name}"`)
      return res
    } catch (err) {
      showToast(`Lỗi tải file: ${err.response?.data?.detail || err.message}`, 'error')
      throw err
    }
  }, [showToast])

  return (
    <div className="flex h-screen overflow-hidden bg-surface-950 text-surface-50">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={newChat}
        onDelete={deleteConversation}
        onUpload={handleUpload}
        dbStatus={dbStatus}
      />

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <ChatArea
          conversation={active}
          loading={loading}
          settings={settings}
          onSend={sendMessage}
          onUpload={handleUpload}
          onClear={clearMessages}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </main>

      {settingsOpen && (
        <SettingsDrawer
          settings={settings}
          onChange={setSettings}
          onClose={() => setSettingsOpen(false)}
          dbStatus={dbStatus}
        />
      )}

      {/* Toast */}
      {toast && (
        <div className={`fixed bottom-6 right-6 z-50 animate-fade-up px-4 py-3 rounded-xl text-sm shadow-lg flex items-center gap-2 ${
          toast.type === 'error'
            ? 'bg-red-500/15 border border-red-500/25 text-red-300'
            : 'bg-emerald-500/15 border border-emerald-500/25 text-emerald-300'
        }`}>
          <span className={`w-2 h-2 rounded-full shrink-0 ${toast.type === 'error' ? 'bg-red-400' : 'bg-emerald-400'}`} />
          {toast.msg}
        </div>
      )}
    </div>
  )
}
