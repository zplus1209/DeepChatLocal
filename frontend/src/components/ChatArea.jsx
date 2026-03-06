import { useEffect, useRef } from 'react'
import { MessageSquareDashed, SlidersHorizontal, Trash2 } from 'lucide-react'
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
  onSend, onUpload, onClear, onOpenSettings
}) {
  const bottomRef = useRef(null)
  const scrollRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages, loading])

  const messages = conversation?.messages || []
  const isEmpty = messages.length === 0

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
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
          {/* Active settings badges */}
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
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {loading && <TypingIndicator />}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput onSend={onSend} onFileUpload={onUpload} loading={loading} />
    </div>
  )
}

function EmptyState({ onSuggest }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center h-full px-6 py-16 gap-8">
      {/* Icon */}
      <div className="w-16 h-16 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center">
        <MessageSquareDashed size={28} className="text-accent/60" />
      </div>

      <div className="text-center space-y-2 max-w-sm">
        <h2 className="text-lg font-semibold text-surface-50">Bắt đầu cuộc trò chuyện</h2>
        <p className="text-sm text-surface-200/50 leading-relaxed">
          Tải tài liệu lên và đặt câu hỏi, hoặc trò chuyện trực tiếp với AI
        </p>
      </div>

      {/* Suggestions */}
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
