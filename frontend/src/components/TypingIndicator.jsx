export default function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-up">
      <div className="w-7 h-7 rounded-full bg-accent/20 ring-1 ring-accent/30 flex items-center justify-center text-xs font-semibold text-accent shrink-0">
        AI
      </div>
      <div className="bg-surface-800 border border-white/5 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1.5">
        {[0, 1, 2].map(i => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-accent/60 animate-pulse-dot"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
      </div>
    </div>
  )
}
