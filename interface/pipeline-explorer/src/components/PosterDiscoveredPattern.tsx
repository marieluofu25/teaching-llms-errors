import { useState } from 'react'
import { P } from '../paths'

type Tag = { label: string }

type Props = {
  headline: string
  body: string
  tags: Tag[]
  caption: string
  empty: string
  showPanel: boolean
}

function OptionalImg({ src, alt }: { src: string; alt: string }) {
  const [ok, setOk] = useState(true)
  if (!ok) return null
  return (
    <img
      src={src}
      alt={alt}
      className="w-full rounded border border-teal-200/80 bg-white"
      onError={() => setOk(false)}
    />
  )
}

export function PosterDiscoveredPattern({ headline, body, tags, caption, empty, showPanel }: Props) {
  if (!showPanel) {
    return <p className="text-[10px] text-slate-500">{empty}</p>
  }

  return (
    <div className="space-y-2">
      <div className="rounded-md border border-teal-300/70 bg-teal-50/50 border-l-[3px] border-l-teal-600 p-2.5">
        <p className="font-mono text-[8px] text-teal-800 mb-1">{headline}</p>
        <p className="text-[10px] text-slate-700 leading-snug">{body}</p>
        <div className="flex flex-wrap gap-1 mt-2">
          {tags.map((t) => (
            <span
              key={t.label}
              className="font-mono text-[8px] px-1.5 py-0.5 rounded-full bg-white/90 border border-teal-200 text-teal-900"
            >
              {t.label}
            </span>
          ))}
        </div>
      </div>

      <OptionalImg src={P.saeGroupCompareViz} alt="High vs low residual SAE group comparison" />

      <p className="text-[9px] text-slate-600 italic leading-snug">{caption}</p>
    </div>
  )
}
