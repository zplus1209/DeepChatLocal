import { useEffect, useState, useCallback } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import InfoPanel from './components/InfoPanel'
import SettingsDrawer from './components/SettingsDrawer'
import { ThemeProvider } from './context/ThemeContext'
import { useChat } from './hooks/useChat'
import { useResize } from './hooks/useResize'
import { api } from './api'

function AppInner() {
  const [dbStatus, setDbStatus] = useState(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [toast, setToast] = useState(null)
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)
  const [sources, setSources] = useState([])

  const leftPanel = useResize(256, 200, 400)
  const rightPanel = useResize(340, 240, 520)

  const {
    conversations, active, activeId,
    setActiveId, loading, settings, setSettings,
    sendMessage, newChat, deleteConversation, clearMessages
  } = useChat()

  useEffect(() => {
    api.health().then(setDbStatus).catch(() => setDbStatus(null))
  }, [])

  const showToast = useCallback((msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3200)
  }, [])

  const handleUpload = useCallback(async (file, onProgress) => {
    try {
      const res = await api.ingestFile(file, onProgress)
      showToast(`Đã nhập ${res.count} đoạn từ "${file.name}"`)
      return res
    } catch (err) {
      showToast(`Lỗi: ${err.response?.data?.detail || err.message}`, 'error')
      throw err
    }
  }, [showToast])

  const handleUploadMany = useCallback(async (files, onProgress) => {
    try {
      const res = await api.ingestFiles(files, onProgress)
      showToast(`Đã nhập ${res.total_ids} đoạn từ ${res.total_files} file`)
      return res
    } catch (err) {
      showToast(`Lỗi: ${err.response?.data?.detail || err.message}`, 'error')
      throw err
    }
  }, [showToast])

  const handleIngestFolder = useCallback(async (folderPath, recursive) => {
    try {
      const res = await api.ingestFolder(folderPath, recursive)
      showToast(`Đã nhập ${res.total_chunks} đoạn từ ${res.total_files} file`)
      return res
    } catch (err) {
      showToast(`Lỗi folder: ${err.response?.data?.detail || err.message}`, 'error')
      throw err
    }
  }, [showToast])

  return (
    <div className="flex h-screen overflow-hidden bg-surface-950 text-surface-50 select-none">

      {/* ── Left panel ── */}
      <div
        className="relative flex-shrink-0 transition-[width] duration-200 ease-out overflow-hidden border-r border-white/5"
        style={{ width: leftOpen ? leftPanel.width : 0 }}
      >
        <div style={{ width: leftPanel.width }} className="h-full">
          <Sidebar
            conversations={conversations}
            activeId={activeId}
            onSelect={setActiveId}
            onNew={newChat}
            onDelete={deleteConversation}
            onUpload={handleUpload}
            onUploadMany={handleUploadMany}
            onIngestFolder={handleIngestFolder}
            dbStatus={dbStatus}
          />
        </div>
        {/* Drag handle */}
        <div
          className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-accent/40 transition-colors z-20"
          onMouseDown={leftPanel.onMouseDown}
        />
      </div>

      {/* Toggle left */}
      <button
        onClick={() => setLeftOpen(v => !v)}
        className="flex-shrink-0 w-4 flex items-center justify-center bg-surface-900 border-r border-white/5 text-surface-200/30 hover:text-surface-200 hover:bg-surface-800 transition-colors z-10"
        title={leftOpen ? 'Ẩn sidebar' : 'Hiện sidebar'}
      >
        {leftOpen ? <ChevronLeft size={10} /> : <ChevronRight size={10} />}
      </button>

      {/* ── Main chat ── */}
      <main className="flex-1 min-w-0 flex flex-col overflow-hidden select-text">
        <ChatArea
          conversation={active}
          loading={loading}
          settings={settings}
          onSend={sendMessage}
          onUpload={handleUpload}
          onClear={clearMessages}
          onOpenSettings={() => setSettingsOpen(true)}
          onSourcesChange={setSources}
        />
      </main>

      {/* Toggle right */}
      <button
        onClick={() => setRightOpen(v => !v)}
        className="flex-shrink-0 w-4 flex items-center justify-center bg-surface-900 border-l border-white/5 text-surface-200/30 hover:text-surface-200 hover:bg-surface-800 transition-colors z-10"
        title={rightOpen ? 'Ẩn panel nguồn' : 'Hiện panel nguồn'}
      >
        {rightOpen ? <ChevronRight size={10} /> : <ChevronLeft size={10} />}
      </button>

      {/* ── Right info panel ── */}
      <div
        className="relative flex-shrink-0 transition-[width] duration-200 ease-out overflow-hidden"
        style={{ width: rightOpen ? rightPanel.width : 0 }}
      >
        {/* Drag handle */}
        <div
          className="absolute top-0 left-0 w-1 h-full cursor-col-resize hover:bg-accent/40 transition-colors z-20"
          onMouseDown={(e) => {
            // Kéo từ cạnh trái → mở rộng sang trái nên delta âm
            const start = e.clientX
            const startW = rightPanel.width
            document.body.style.cursor = 'col-resize'
            document.body.style.userSelect = 'none'
            const move = (ev) => {
              const delta = start - ev.clientX
              const newW = Math.min(520, Math.max(240, startW + delta))
              // gọi setWidth của rightPanel hook không exposed trực tiếp
              // workaround: dùng css width trên element
            }
            window.addEventListener('mousemove', move)
            window.addEventListener('mouseup', () => {
              window.removeEventListener('mousemove', move)
              document.body.style.cursor = ''
              document.body.style.userSelect = ''
            }, { once: true })
          }}
        />
        <div style={{ width: rightPanel.width }} className="h-full">
          <InfoPanel sources={sources} />
        </div>
      </div>

      {/* Settings drawer */}
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
        <div className={`fixed bottom-6 right-6 z-50 animate-fade-up px-4 py-2.5 rounded-xl text-sm font-medium shadow-xl flex items-center gap-2 ${
          toast.type === 'error'
            ? 'bg-red-950/90 text-red-300 border border-red-500/20'
            : 'bg-surface-900 text-surface-50 border border-white/10'
        }`}>
          {toast.msg}
        </div>
      )}
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <AppInner />
    </ThemeProvider>
  )
}