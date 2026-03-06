import { useState } from 'react'
import {
  MessageSquare, Plus, Trash2, ChevronRight,
  Database, Upload, Search, X, Layers
} from 'lucide-react'

const timeAgo = (ts) => {
  const d = Date.now() - ts
  if (d < 60000) return 'vừa xong'
  if (d < 3600000) return `${Math.floor(d / 60000)} phút`
  if (d < 86400000) return `${Math.floor(d / 3600000)} giờ`
  return `${Math.floor(d / 86400000)} ngày`
}

export default function Sidebar({
  conversations, activeId, onSelect, onNew, onDelete,
  onUpload, dbStatus
}) {
  const [tab, setTab] = useState('chats')  // 'chats' | 'files'
  const [search, setSearch] = useState('')

  const filtered = conversations.filter(c =>
    c.title.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <aside className="w-64 shrink-0 flex flex-col h-full bg-surface-900 border-r border-white/5">
      {/* Logo */}
      <div className="px-4 py-4 flex items-center gap-2.5 border-b border-white/5">
        <div className="w-7 h-7 rounded-lg bg-accent/20 flex items-center justify-center shrink-0">
          <Layers size={14} className="text-accent" />
        </div>
        <span className="font-semibold text-sm tracking-tight text-white">DeepChat Local</span>
        {dbStatus && (
          <span className="ml-auto w-2 h-2 rounded-full bg-emerald-400 shrink-0" title={dbStatus.db_type} />
        )}
      </div>

      {/* Tabs */}
      <div className="flex px-2 pt-2 gap-1">
        {[
          { id: 'chats', icon: MessageSquare, label: 'Chats' },
          { id: 'files', icon: Database, label: 'Files' },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
              tab === t.id ? 'bg-white/8 text-white' : 'text-surface-200 hover:text-white'
            }`}
          >
            <t.icon size={12} />
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden flex flex-col min-h-0 px-2 pt-2">
        {tab === 'chats' && (
          <>
            {/* New chat */}
            <button onClick={onNew} className="btn-ghost w-full justify-start mb-2 text-xs">
              <Plus size={14} className="text-accent" />
              Cuộc trò chuyện mới
            </button>

            {/* Search */}
            <div className="relative mb-2">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-surface-200/50" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Tìm kiếm…"
                className="input-base w-full pl-7 text-xs py-1.5"
              />
              {search && (
                <button onClick={() => setSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-surface-200/50 hover:text-white">
                  <X size={12} />
                </button>
              )}
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto space-y-0.5">
              {filtered.map(c => (
                <ConvItem
                  key={c.id}
                  conv={c}
                  active={c.id === activeId}
                  onSelect={() => onSelect(c.id)}
                  onDelete={() => onDelete(c.id)}
                />
              ))}
              {filtered.length === 0 && (
                <p className="text-center text-xs text-surface-200/40 py-8">Không tìm thấy</p>
              )}
            </div>
          </>
        )}

        {tab === 'files' && (
          <FilePanel onUpload={onUpload} />
        )}
      </div>

      {/* DB status footer */}
      {dbStatus && (
        <div className="px-3 py-2.5 border-t border-white/5 flex items-center gap-2">
          <div className="flex-1 min-w-0">
            <p className="text-xs text-surface-200/60 truncate font-mono">
              {dbStatus.db_type} · {dbStatus.retrieval_mode}
            </p>
          </div>
        </div>
      )}
    </aside>
  )
}

function ConvItem({ conv, active, onSelect, onDelete }) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      className={`group relative flex items-center rounded-md cursor-pointer transition-colors px-2 py-2 ${
        active ? 'bg-white/8 text-white' : 'text-surface-200 hover:bg-white/4 hover:text-white'
      }`}
      onClick={onSelect}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <MessageSquare size={12} className={`shrink-0 mr-2 ${active ? 'text-accent' : 'text-surface-200/50'}`} />
      <div className="flex-1 min-w-0">
        <p className="text-xs truncate font-medium">{conv.title}</p>
        <p className="text-[10px] text-surface-200/40">{timeAgo(conv.createdAt)}</p>
      </div>
      {hovered && (
        <button
          onClick={e => { e.stopPropagation(); onDelete() }}
          className="shrink-0 p-1 rounded hover:bg-white/10 text-surface-200/50 hover:text-red-400 transition-colors"
        >
          <Trash2 size={11} />
        </button>
      )}
      {active && !hovered && <ChevronRight size={11} className="shrink-0 text-accent/60" />}
    </div>
  )
}

function FilePanel({ onUpload }) {
  const [files, setFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)

  const handleDrop = async (e) => {
    e.preventDefault()
    const dropped = Array.from(e.dataTransfer.files)
    await uploadFiles(dropped)
  }

  const uploadFiles = async (fileList) => {
    setUploading(true)
    for (const f of fileList) {
      try {
        await onUpload(f, p => setProgress(p))
        setFiles(prev => [...prev, { name: f.name, size: f.size, uploadedAt: Date.now() }])
      } catch {
        // handled upstream
      }
    }
    setUploading(false)
    setProgress(0)
  }

  const handleInput = async (e) => {
    const picked = Array.from(e.target.files)
    await uploadFiles(picked)
    e.target.value = ''
  }

  return (
    <div className="flex-1 flex flex-col gap-2 overflow-hidden">
      {/* Drop zone */}
      <label
        className={`rounded-lg border-2 border-dashed p-4 flex flex-col items-center gap-2 cursor-pointer transition-colors ${
          uploading ? 'border-accent/60 bg-accent/5' : 'border-white/10 hover:border-accent/40 hover:bg-white/3'
        }`}
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
      >
        <Upload size={18} className={uploading ? 'text-accent' : 'text-surface-200/40'} />
        <p className="text-xs text-surface-200/60 text-center">
          {uploading ? `Đang tải… ${progress}%` : 'Kéo thả hoặc click để tải file'}
        </p>
        <p className="text-[10px] text-surface-200/30">PDF, DOCX, TXT, MD</p>
        <input type="file" className="hidden" multiple accept=".pdf,.docx,.txt,.md" onChange={handleInput} />
      </label>

      {/* File list */}
      <div className="flex-1 overflow-y-auto space-y-1">
        {files.map((f, i) => (
          <div key={i} className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-white/3 group">
            <div className="w-6 h-6 rounded bg-accent/15 flex items-center justify-center shrink-0">
              <Database size={10} className="text-accent" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs truncate text-surface-50">{f.name}</p>
              <p className="text-[10px] text-surface-200/40">{(f.size / 1024).toFixed(1)} KB</p>
            </div>
          </div>
        ))}
        {files.length === 0 && (
          <p className="text-center text-xs text-surface-200/30 py-6">Chưa có file nào</p>
        )}
      </div>
    </div>
  )
}
