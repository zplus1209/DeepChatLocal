import { useEffect, useMemo, useRef, useState } from 'react'
import { MessageSquareDashed, SlidersHorizontal, Trash2, FileText } from 'lucide-react'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'
import ChatInput from './ChatInput'

const SUGGESTIONS = [
  'Tóm tắt nội dung tài liệu đã tải lên',
  'Các điểm chính trong văn bản là gì?',
  'Phân tích và so sánh các thông tin',
  'Giải thích chi tiết hơn về chủ đề này',
]

export default function ChatArea({
  conversation, loading, settings,
  onSend, onUpload, onClear, onOpenSettings,
}) {
  const bottomRef = useRef(null)
  const scrollRef = useRef(null)
  const [inspectSources, setInspectSources] = useState([])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages, loading])

  const messages = conversation?.messages || []
  const isEmpty = messages.length === 0

  const latestAssistantSources = useMemo(() => {
    const assistantMsgs = [...messages].reverse().filter(m => m.role === 'assistant' && (m.sources?.length || 0) > 0)
    return assistantMsgs[0]?.sources || []
  }, [messages])

  useEffect(() => {
    if (latestAssistantSources.length > 0) {
      setInspectSources(latestAssistantSources)
    }
  }, [latestAssistantSources])

  return (
    <div className="flex-1 flex min-h-0 overflow-hidden">
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden border-r border-white/5">
        {/* Topbar */}
        <header className="flex items-center justify-between px-5 py-3.5 border-b border-white/5 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <h1 className="font-semibold text-sm truncate text-surface-50">
              {conversation?.title || 'Cuộc trò chuyện mới'}
            </h1>
            {messages.length > 0 && (
              <span className="text-[10px] text-surface-200/30 shrink-0">
                {messages.length} tin nhắn
              </span>
            )}
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <div className="flex gap-1 mr-2">
              {settings.useHybrid && <span className="tag">hybrid</span>}
              {settings.useRerank && <span className="tag">rerank</span>}
              {settings.useReflection && <span className="tag">reflect</span>}
            </div>

            {messages.length > 0 && (
              <button onClick={onClear} className="btn-ghost text-xs py-1.5 px-2 text-surface-200/50">
                <Trash2 size={13} />
              </button>
            )}
            <button onClick={onOpenSettings} className="btn-ghost text-xs py-1.5">
              <SlidersHorizontal size={14} />
              Cài đặt
            </button>
          </div>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {isEmpty ? (
            <EmptyState onSuggest={onSend} />
          ) : (
            <div className="max-w-3xl mx-auto px-5 py-6 space-y-5">
              {messages.map(msg => (
                <MessageBubble key={msg.id} message={msg} onInspectSources={setInspectSources} />
              ))}
              {loading && <TypingIndicator />}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <ChatInput onSend={onSend} onFileUpload={onUpload} loading={loading} />
      </div>

      <aside className="w-[360px] shrink-0 bg-surface-900/70">
        <SourceInspector sources={inspectSources} />
      </aside>
    </div>
  )
}

function SourceInspector({ sources }) {
  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
        <FileText size={14} className="text-accent" />
        <h3 className="text-sm font-medium">Nguồn tham khảo</h3>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {sources?.length ? sources.map((s, i) => (
          <div key={`${s.metadata?.chunk_id || i}`} className="rounded-xl border border-white/10 bg-surface-900 p-3 text-xs space-y-1.5">
            <div className="text-surface-200/60 space-x-2">
              <span className="tag">#{i + 1}</span>
              <span>{s.metadata?.source || 'unknown'}</span>
            </div>
            <div className="text-surface-200/45">
              trang: <b>{s.metadata?.page ?? '-'}</b> • dòng: <b>{s.metadata?.line_start ?? '-'}</b>-<b>{s.metadata?.line_end ?? '-'}</b>
            </div>
            <div className="text-surface-200/45">
              loại: <b>{s.metadata?.chunk_type || '-'}</b>
            </div>
            <div className="text-surface-200/45 break-all">
              chunk_id: <b>{s.metadata?.chunk_id || '-'}</b>
            </div>
            <div className="text-surface-300/80 leading-relaxed">{(s.content || '').slice(0, 220)}{(s.content || '').length > 220 ? '…' : ''}</div>
          </div>
        )) : (
          <p className="text-xs text-surface-200/40">Chưa có nguồn. Hãy hỏi câu có bật RAG hoặc chọn một phản hồi có nguồn.</p>
        )}
      </div>
    </div>
  )
}

function EmptyState({ onSuggest }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center h-full px-6 py-16 gap-8">
      <div className="w-16 h-16 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center">
        <MessageSquareDashed size={28} className="text-accent/60" />
      </div>

      <div className="text-center space-y-2 max-w-sm">
        <h2 className="text-lg font-semibold text-surface-50">Bắt đầu cuộc trò chuyện</h2>
        <p className="text-sm text-surface-200/50 leading-relaxed">
          Tải tài liệu lên và đặt câu hỏi, hoặc trò chuyện trực tiếp với AI
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2 max-w-lg w-full">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            onClick={() => onSuggest(s)}
            className="text-left px-3.5 py-3 rounded-xl bg-surface-900 border border-white/5 hover:border-accent/30 hover:bg-accent/5 text-xs text-surface-200/70 hover:text-surface-50 transition-all duration-150 leading-relaxed"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}
