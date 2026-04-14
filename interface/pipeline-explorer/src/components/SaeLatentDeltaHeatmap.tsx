import { useMemo } from 'react'
import type { SubjectLatentDeltaHeatmap as HeatmapData } from '../types'

function heatScale(values: (number | null)[][]) {
  const flat = values.flat().filter((x): x is number => x != null && Number.isFinite(x))
  if (!flat.length) return 1e-6
  const abs = flat.map(Math.abs).sort((a, b) => a - b)
  const i = Math.min(abs.length - 1, Math.floor(abs.length * 0.98))
  return Math.max(abs[i] ?? 1e-6, 1e-9)
}

function cellBg(v: number | null, scale: number): string {
  if (v == null || !Number.isFinite(v)) return '#e5e7eb'
  const u = Math.max(-1, Math.min(1, v / scale))
  if (u <= 0) {
    const t = 1 + u
    const r = Math.round(37 + (255 - 37) * t)
    const g = Math.round(99 + (255 - 99) * t)
    const b = Math.round(235 + (255 - 235) * t)
    return `rgb(${r},${g},${b})`
  }
  const t = u
  const r = 255
  const g = Math.round(255 - (255 - 220) * t)
  const b = Math.round(255 - (255 - 38) * t)
  return `rgb(${r},${g},${b})`
}

function textColor(v: number | null): string {
  if (v == null || !Number.isFinite(v)) return '#64748b'
  const a = Math.abs(v)
  return a > 0.35 ? '#ffffff' : '#0f172a'
}

type Props = {
  data: HeatmapData
  legendLow: string
  legendHigh: string
  nullLabel: string
}

export function SaeLatentDeltaHeatmap({ data, legendLow, legendHigh, nullLabel }: Props) {
  const scale = useMemo(() => heatScale(data.values), [data.values])

  return (
    <div className="w-full min-w-0">
      <div className="overflow-x-auto rounded border border-slate-200">
        <div
          className="inline-grid gap-px bg-slate-300 p-px text-[7px] leading-tight"
          style={{
            gridTemplateColumns: `minmax(5.5rem,7rem) repeat(${data.latent_ids.length}, minmax(1.65rem, 1fr))`,
          }}
        >
          <div className="bg-slate-100 px-1 py-1 font-mono font-semibold text-slate-600 flex items-end">
            subject \ latent
          </div>
          {data.latent_ids.map((lid) => (
            <div
              key={lid}
              className="bg-slate-100 px-0.5 py-1 font-mono text-[6.5px] text-slate-700 text-center break-all"
              title={`latent_id ${lid}`}
            >
              {lid}
            </div>
          ))}
          {data.subjects.map((subj, i) => (
            <div key={subj} className="contents">
              <div
                className="bg-white px-1 py-0.5 font-mono text-[6.5px] text-slate-800 text-right truncate"
                title={subj}
              >
                {subj.replace(/_/g, ' ')}
              </div>
              {data.latent_ids.map((lid, j) => {
                const v = data.values[i]?.[j] ?? null
                const nh = data.n_high?.[i]?.[j]
                const nl = data.n_low?.[i]?.[j]
                const tip =
                  v == null
                    ? `${nullLabel}${nh != null && nl != null ? ` (n_high=${nh}, n_low=${nl})` : ''}`
                    : `Δ=${v.toFixed(3)} · latent ${lid}${nh != null && nl != null ? ` · n_high=${nh}, n_low=${nl}` : ''}`
                return (
                  <div
                    key={`${subj}-${lid}`}
                    className="min-h-[1.1rem] flex items-center justify-center font-mono tabular-nums"
                    style={{ backgroundColor: cellBg(v, scale), color: textColor(v) }}
                    title={tip}
                  >
                    {v == null ? '·' : v > 0 ? '+' : ''}
                    {v == null ? '' : v.toFixed(2)}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 mt-2">
        <div
          className="h-2 w-28 rounded border border-slate-200"
          style={{
            background: 'linear-gradient(90deg, rgb(37,99,235), rgb(255,255,255), rgb(255,38,38))',
          }}
        />
        <span className="font-mono text-[8px] text-slate-600">
          {legendLow} · 0 · {legendHigh} (±{scale.toFixed(2)})
        </span>
      </div>
    </div>
  )
}
