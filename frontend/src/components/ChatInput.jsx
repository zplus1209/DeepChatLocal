import { useState, useRef } from 'react'
import { Send, Paperclip, Square } from 'lucide-react'

export default function ChatInput({ onSend, onFileUpload, loading }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)
  const fileRef = useRef(null)

  const submit = () => {
    const val = text.trim()
    if (!val || loading) return
    onSend(val)
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const handleInput = (e) => {
    setText(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 160) + 'px'
    }
  }

  const handleFile = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    await onFileUpload(file)
    e.target.value = ''
  }

  return (
    <div className="p-4 border-t border-white/5 bg-surface-950/80 backdrop-blur-sm">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-end gap-2 bg-surface-900 border border-white/8 rounded-2xl px-3 py-2 focus-within:border-accent/40 transition-colors">
          {/* Attach */}
          <button
            onClick={() => fileRef.current?.click()}
            className="shrink-0 p-1.5 rounded-lg text-surface-200/40 hover:text-surface-200 hover:bg-white/5 transition-colors mb-0.5"
            title="Đính kèm file"
          >
            <Paperclip size={16} />
          </button>
          <input ref={fileRef} type="file" className="hidden" accept=".pdf,.docx,.txt,.md" onChange={handleFile} />

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKey}
            placeholder="Nhập câu hỏi… (Enter gửi, Shift+Enter xuống dòng)"
            rows={1}
            className="flex-1 bg-transparent resize-none text-sm text-surface-50 placeholder:text-surface-200/30 focus:outline-none py-1.5 max-h-40 leading-relaxed"
          />

          {/* Send */}
          <button
            onClick={submit}
            disabled={!text.trim() && !loading}
            className={`shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-all mb-0.5 ${
              text.trim() && !loading
                ? 'bg-accent hover:bg-accent-hover text-white active:scale-90'
                : loading
                  ? 'bg-surface-800 text-surface-200/50'
                  : 'bg-surface-800 text-surface-200/20 cursor-not-allowed'
            }`}
          >
            {loading ? <Square size={12} fill="currentColor" /> : <Send size={13} />}
          </button>
        </div>

        <p className="text-center text-[10px] text-surface-200/25 mt-2">
          DeepChat Local — AI có thể mắc lỗi, hãy kiểm tra thông tin quan trọng
        </p>
      </div>
    </div>
  )
}
