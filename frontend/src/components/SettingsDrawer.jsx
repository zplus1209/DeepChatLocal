import { X, Info } from 'lucide-react'

const Toggle = ({ label, desc, value, onChange }) => (
  <label className="flex items-start gap-3 cursor-pointer group">
    <div className="mt-0.5 shrink-0">
      <div
        className={`w-9 h-5 rounded-full transition-colors relative ${value ? 'bg-accent' : 'bg-surface-800 border border-white/10'}`}
        onClick={() => onChange(!value)}
      >
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${value ? 'translate-x-4' : 'translate-x-0.5'}`} />
      </div>
    </div>
    <div>
      <p className="text-sm font-medium text-surface-50">{label}</p>
      {desc && <p className="text-xs text-surface-200/50 mt-0.5 leading-relaxed">{desc}</p>}
    </div>
  </label>
)

export default function SettingsDrawer({ settings, onChange, onClose, dbStatus }) {
  const set = (key) => (val) => onChange({ ...settings, [key]: val })

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />

      <aside className="relative w-80 h-full bg-surface-900 border-l border-white/5 flex flex-col animate-slide-in shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/5">
          <h2 className="font-semibold text-sm">Cài đặt RAG</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-surface-200/60 hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
          {/* DB info */}
          {dbStatus && (
            <section>
              <SectionTitle>Kết nối</SectionTitle>
              <div className="bg-surface-800/60 rounded-xl p-3 space-y-2 text-xs font-mono border border-white/5">
                <Row label="DB" value={dbStatus.db_type} />
                <Row label="Mode" value={dbStatus.retrieval_mode} />
                <Row label="Embedding" value={dbStatus.embedding_model?.split('/').pop()} />
              </div>
            </section>
          )}

          {/* Retrieval */}
          <section className="space-y-4">
            <SectionTitle>Tìm kiếm</SectionTitle>

            <Toggle
              label="Dùng RAG"
              desc="Tìm kiếm tài liệu liên quan trước khi trả lời"
              value={settings.useRag}
              onChange={set('useRag')}
            />
            <Toggle
              label="Hybrid Search"
              desc="Kết hợp vector search + keyword search, tăng độ bao phủ"
              value={settings.useHybrid}
              onChange={set('useHybrid')}
            />
            <Toggle
              label="Rerank"
              desc="Dùng CrossEncoder để xếp hạng lại kết quả tìm được"
              value={settings.useRerank}
              onChange={set('useRerank')}
            />
          </section>

          {/* Generation */}
          <section className="space-y-4">
            <SectionTitle>Sinh văn bản</SectionTitle>

            <Toggle
              label="Reflection"
              desc="Tự động cải thiện câu hỏi dựa trên lịch sử hội thoại"
              value={settings.useReflection}
              onChange={set('useReflection')}
            />
          </section>

          {/* Note */}
          <div className="flex gap-2 bg-accent/5 border border-accent/15 rounded-xl p-3 text-xs text-surface-200/60">
            <Info size={13} className="text-accent/70 mt-0.5 shrink-0" />
            <p>Hybrid Search và Rerank yêu cầu backend đã cấu hình tương ứng trong <code className="font-mono text-accent/80">.env</code></p>
          </div>
        </div>
      </aside>
    </div>
  )
}

const SectionTitle = ({ children }) => (
  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-surface-200/40 mb-3">{children}</h3>
)

const Row = ({ label, value }) => (
  <div className="flex justify-between text-surface-200/60">
    <span className="text-surface-200/40">{label}</span>
    <span className="text-surface-50">{value || '—'}</span>
  </div>
)
