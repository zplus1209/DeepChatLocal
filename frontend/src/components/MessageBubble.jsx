import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useState } from 'react'
import { ChevronDown, ChevronUp, FileText, AlertCircle } from 'lucide-react'

const Avatar = ({ role }) => (
  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 ${
    role === 'user'
      ? 'bg-surface-800 text-surface-50 ring-1 ring-white/10'
      : 'bg-accent/20 text-accent ring-1 ring-accent/30'
  }`}>
    {role === 'user' ? 'U' : 'AI'}
  </div>
)

export default function MessageBubble({ message, onInspectSources }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const hasSources = message.sources?.length > 0

  return (
    <div className={`flex gap-3 group animate-fade-up ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
      <Avatar role={message.role} />

      <div className={`flex flex-col gap-1.5 max-w-[75%] ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
        {/* Bubble */}
        <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          message.role === 'user'
            ? 'bg-accent text-white rounded-tr-sm'
            : message.error
              ? 'bg-red-500/10 border border-red-500/20 text-red-300 rounded-tl-sm'
              : 'bg-surface-800 text-surface-50 rounded-tl-sm border border-white/5'
        }`}>
          {message.error && (
            <div className="flex items-center gap-1.5 mb-1 text-red-400">
              <AlertCircle size={13} />
              <span className="text-xs font-medium">Lỗi</span>
            </div>
          )}
          {message.role === 'user' ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MarkdownContent content={message.content} />
          )}
        </div>

        {/* Sources toggle */}
        {hasSources && (
          <div className="w-full">
            <button
              onClick={() => onInspectSources && onInspectSources(message.sources)}
              className="mb-1 text-[11px] text-accent/70 hover:text-accent"
            >
              Xem ở panel bên phải
            </button>
            <button
              onClick={() => setSourcesOpen(v => !v)}
              className="flex items-center gap-1.5 text-xs text-surface-200/50 hover:text-surface-200 transition-colors py-1"
            >
              <FileText size={11} />
              {message.sources.length} nguồn tham khảo
              {sourcesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>

            {sourcesOpen && (
              <div className="mt-1 space-y-1.5 animate-fade-up">
                {message.sources.map((s, i) => (
                  <SourceCard key={i} source={s} index={i + 1} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function MarkdownContent({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        h1: ({ children }) => <h1 className="text-base font-semibold mb-2 mt-3 first:mt-0">{children}</h1>,
        h2: ({ children }) => <h2 className="text-sm font-semibold mb-1.5 mt-3 first:mt-0">{children}</h2>,
        h3: ({ children }) => <h3 className="text-sm font-medium mb-1 mt-2 first:mt-0">{children}</h3>,
        ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5 pl-1">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5 pl-1">{children}</ol>,
        li: ({ children }) => <li className="text-sm">{children}</li>,
        code: ({ inline, children }) =>
          inline
            ? <code className="font-mono text-xs bg-white/10 text-accent px-1.5 py-0.5 rounded">{children}</code>
            : <pre className="bg-black/30 rounded-lg p-3 overflow-x-auto my-2 border border-white/5">
                <code className="font-mono text-xs text-surface-100">{children}</code>
              </pre>,
        blockquote: ({ children }) =>
          <blockquote className="border-l-2 border-accent/50 pl-3 my-2 text-surface-200/70">{children}</blockquote>,
        table: ({ children }) =>
          <div className="overflow-x-auto my-2">
            <table className="min-w-full text-xs border-collapse">{children}</table>
          </div>,
        th: ({ children }) => <th className="border border-white/10 bg-white/5 px-2 py-1 text-left font-medium">{children}</th>,
        td: ({ children }) => <td className="border border-white/5 px-2 py-1">{children}</td>,
        a: ({ href, children }) =>
          <a href={href} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-2 hover:text-accent-hover">{children}</a>,
        strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false)
  const preview = source.content?.slice(0, 180)
  const full = source.content || ''
  const needsExpand = full.length > 180

  return (
    <div className="bg-surface-900/80 border border-white/5 rounded-xl p-3 text-xs space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="tag">#{index}</span>
        {source.metadata?.source && (
          <span className="text-surface-200/50 truncate max-w-[200px]">
            {source.metadata.source}
          </span>
        )}
        {source.score !== undefined && (
          <span className="ml-auto text-surface-200/40 font-mono">
            {(source.score * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <p className="text-surface-200/70 leading-relaxed">
        {expanded ? full : preview}
        {needsExpand && !expanded && '…'}
      </p>

      {needsExpand && (
        <button
          onClick={() => setExpanded(v => !v)}
          className="text-accent/70 hover:text-accent transition-colors text-[11px]"
        >
          {expanded ? 'Thu gọn' : 'Xem thêm'}
        </button>
      )}
    </div>
  )
}
