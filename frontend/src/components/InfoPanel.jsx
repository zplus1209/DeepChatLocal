import { useEffect, useRef, useState } from 'react'
import { FileText, ChevronDown, ChevronRight, Loader, AlertCircle } from 'lucide-react'

const ACCENT_COLORS = [
  'rgba(108,99,255,0.35)',   // tím
  'rgba(34,197,94,0.30)',    // xanh lá
  'rgba(249,115,22,0.32)',   // cam
  'rgba(236,72,153,0.32)',   // hồng
]

export default function InfoPanel({ sources = [] }) {
  const [active, setActive] = useState(0)

  useEffect(() => { setActive(0) }, [sources])

  if (sources.length === 0) {
    return (
      <div className="h-full flex flex-col bg-surface-900 border-l border-white/5">
        <Header count={0} />
        <div className="flex-1 flex flex-col items-center justify-center gap-3 px-6 text-center">
          <FileText size={28} className="text-surface-200/20" />
          <p className="text-xs text-surface-200/40 leading-relaxed">
            Gửi câu hỏi với RAG bật để xem trang tài liệu gốc được dùng.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-surface-900 border-l border-white/5 overflow-hidden">
      <Header count={sources.length} />

      {/* Source tabs */}
      <div className="flex gap-1 px-3 pt-2 pb-1 overflow-x-auto shrink-0 scrollbar-none">
        {sources.map((s, i) => (
          <button
            key={i}
            onClick={() => setActive(i)}
            style={{ borderColor: active === i ? ACCENT_COLORS[i % ACCENT_COLORS.length].replace('0.3', '0.8') : 'transparent' }}
            className={`shrink-0 px-2.5 py-1 rounded-md text-[11px] font-medium border transition-all duration-150 ${
              active === i
                ? 'bg-accent/15 text-white'
                : 'text-surface-200/50 hover:text-surface-200 hover:bg-white/5 border-transparent'
            }`}
          >
            #{i + 1}
          </button>
        ))}
      </div>

      {/* Active source viewer */}
      <div className="flex-1 overflow-y-auto">
        {sources[active] && (
          <SourceViewer
            source={sources[active]}
            index={active}
            color={ACCENT_COLORS[active % ACCENT_COLORS.length]}
          />
        )}
      </div>
    </div>
  )
}

function Header({ count }) {
  return (
    <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5 shrink-0">
      <FileText size={13} className="text-accent" />
      <span className="text-sm font-medium text-surface-50">Nguồn tham khảo</span>
      {count > 0 && (
        <span className="ml-auto text-[11px] font-mono bg-accent/15 text-accent px-2 py-0.5 rounded-full">
          {count}
        </span>
      )}
    </div>
  )
}

function SourceViewer({ source, index, color }) {
  const meta = source.metadata || {}
  const filename = meta.source || ''
  const page = meta.page ?? 1
  const bbox = meta.bbox          // [x1, y1, x2, y2] in PDF pts
  const pageW = meta.page_width   // PDF page width in pts
  const pageH = meta.page_height  // PDF page height in pts
  const hasBbox = bbox && pageW && pageH

  const [showMeta, setShowMeta] = useState(false)
  const imgRef = useRef(null)
  const [imgSize, setImgSize] = useState(null)

  // Khi image load xong, lấy kích thước thực tế trên màn hình
  const onImgLoad = (e) => {
    setImgSize({ w: e.target.clientWidth, h: e.target.clientHeight })
  }

  // Scale bbox từ PDF coordinate → pixel trên màn hình
  const scaledBbox = hasBbox && imgSize
    ? {
        left:   (bbox[0] / pageW) * imgSize.w,
        top:    (bbox[1] / pageH) * imgSize.h,
        width:  ((bbox[2] - bbox[0]) / pageW) * imgSize.w,
        height: ((bbox[3] - bbox[1]) / pageH) * imgSize.h,
      }
    : null

  const isPdf = filename.toLowerCase().endsWith('.pdf')
  const pageImageUrl = isPdf && filename
    ? `/api/v1/page-image?filename=${encodeURIComponent(filename)}&page=${page}&dpi=150`
    : null

  return (
    <div className="flex flex-col gap-3 p-3">
      {/* File + page badge */}
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className="text-[11px] font-medium px-2 py-0.5 rounded-md"
          style={{ background: color.replace('0.3', '0.15'), color: 'rgba(255,255,255,0.85)' }}
        >
          #{index + 1}
        </span>
        <span className="text-xs text-surface-200/60 truncate flex-1" title={filename}>
          {filename || 'Không rõ nguồn'}
        </span>
        <span className="text-[10px] text-surface-200/40 shrink-0">tr. {page}</span>
      </div>

      {/* PDF page render với bbox highlight */}
      {pageImageUrl ? (
        <PageImageWithBbox
          url={pageImageUrl}
          bbox={scaledBbox}
          color={color}
          onLoad={onImgLoad}
          imgRef={imgRef}
        />
      ) : (
        <div className="text-xs text-surface-200/40 italic">
          (Chỉ hỗ trợ xem trang với file PDF)
        </div>
      )}

      {/* Toggle metadata */}
      <button
        onClick={() => setShowMeta(v => !v)}
        className="flex items-center gap-1.5 text-[11px] text-surface-200/50 hover:text-surface-200 transition-colors"
      >
        {showMeta ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        Chi tiết chunk
      </button>

      {showMeta && (
        <div className="bg-surface-950/60 rounded-lg border border-white/5 p-2.5 text-[11px] space-y-1.5">
          <MetaRow k="Loại" v={meta.chunk_type} />
          <MetaRow k="Dòng" v={meta.line_start != null ? `${meta.line_start}–${meta.line_end}` : null} />
          <MetaRow k="Page W×H" v={pageW && pageH ? `${pageW}×${pageH} pt` : null} />
          <MetaRow k="BBox" v={bbox ? `[${bbox.join(', ')}]` : null} mono />
          <MetaRow k="Chunk ID" v={meta.chunk_id} mono />
        </div>
      )}

      {/* Content preview */}
      <ContentPreview content={source.content} />
    </div>
  )
}

function PageImageWithBbox({ url, bbox, color, onLoad, imgRef }) {
  const [status, setStatus] = useState('loading') // loading | ok | error

  return (
    <div className="relative rounded-lg overflow-hidden border border-white/8 bg-black/30">
      {status === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <Loader size={18} className="text-accent animate-spin" />
        </div>
      )}
      {status === 'error' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 z-10">
          <AlertCircle size={18} className="text-red-400/70" />
          <span className="text-[11px] text-surface-200/40">Không tải được trang</span>
        </div>
      )}

      {/* Ảnh trang PDF */}
      <img
        ref={imgRef}
        src={url}
        alt="Trang tài liệu"
        className={`w-full block transition-opacity duration-200 ${status !== 'ok' ? 'opacity-0' : 'opacity-100'}`}
        onLoad={(e) => { setStatus('ok'); onLoad(e) }}
        onError={() => setStatus('error')}
      />

      {/* BBox highlight overlay */}
      {status === 'ok' && bbox && (
        <div
          className="absolute pointer-events-none rounded-sm"
          style={{
            left:   bbox.left,
            top:    bbox.top,
            width:  bbox.width,
            height: bbox.height,
            background: color,
            border: `2px solid ${color.replace('0.3', '0.9')}`,
            boxShadow: `0 0 0 1px ${color.replace('0.3', '0.3')}`,
            transition: 'all 0.15s ease',
          }}
        />
      )}
    </div>
  )
}

function ContentPreview({ content }) {
  const [expanded, setExpanded] = useState(false)
  if (!content) return null
  const preview = content.slice(0, 240)
  const full = content
  const long = full.length > 240

  return (
    <div className="bg-surface-950/50 rounded-lg border border-white/5 p-2.5">
      <p className="text-[11px] text-surface-200/70 leading-relaxed whitespace-pre-wrap">
        {expanded ? full : preview}
        {long && !expanded ? '…' : ''}
      </p>
      {long && (
        <button
          onClick={() => setExpanded(v => !v)}
          className="mt-1.5 text-[10px] text-accent/70 hover:text-accent transition-colors"
        >
          {expanded ? 'Thu gọn' : 'Xem thêm'}
        </button>
      )}
    </div>
  )
}

function MetaRow({ k, v, mono = false }) {
  if (!v) return null
  return (
    <div className="flex gap-2">
      <span className="text-surface-200/35 w-16 shrink-0">{k}</span>
      <span className={`text-surface-200/70 break-all ${mono ? 'font-mono text-[10px]' : ''}`}>{v}</span>
    </div>
  )
}