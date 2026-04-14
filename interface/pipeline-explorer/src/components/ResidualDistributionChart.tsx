import { useMemo } from 'react'
import type { ResidualHistogram } from '../types'

type Props = {
  data: ResidualHistogram
  title: string
  caption: string
  axisLabel: string
  labelQ25: string
  labelQ75: string
}

function pickSeries(data: ResidualHistogram) {
  const z = data.zoom
  if (z?.counts?.length && z.edges && z.edges.length > 1 && z.min != null && z.max != null) {
    return {
      counts: z.counts,
      edges: z.edges,
      lo: z.min,
      hi: z.max,
      percentileWindow: z.percentile_window as [number, number] | undefined,
    }
  }
  const counts = data.counts ?? []
  const edges = data.edges ?? []
  return {
    counts,
    edges,
    lo: data.min ?? 0,
    hi: data.max ?? 1,
    percentileWindow: undefined,
  }
}

export function ResidualDistributionChart({ data, title, caption, axisLabel, labelQ25, labelQ75 }: Props) {
  const { bars, p25, p75, maxC, lo, hi, linesOverlap, q25, q75 } = useMemo(() => {
    const { counts, edges, lo, hi, percentileWindow } = pickSeries(data)
    const span = hi - lo || 1e-12
    const q25 = data.quantiles?.q25 ?? 0
    const q75 = data.quantiles?.q75 ?? 0
    const maxC = Math.max(...counts, 1)
    const n = Math.min(counts.length, Math.max(0, edges.length - 1))
    const bars: { c: number; h: number; tone: 'high' | 'low' | 'mid' }[] = []
    for (let i = 0; i < n; i++) {
      const a = edges[i] ?? lo
      const b = edges[i + 1] ?? hi
      const center = (a + b) / 2
      let tone: 'high' | 'low' | 'mid' = 'mid'
      if (center >= q75) tone = 'high'
      else if (center <= q25) tone = 'low'
      const c = counts[i] ?? 0
      const h = 100 * Math.sqrt(c / maxC)
      bars.push({ c, h, tone })
    }
    const p25 = ((q25 - lo) / span) * 100
    const p75 = ((q75 - lo) / span) * 100
    const linesOverlap = Math.abs(p75 - p25) < 1.2
    return {
      bars,
      p25: Math.min(100, Math.max(0, p25)),
      p75: Math.min(100, Math.max(0, p75)),
      maxC,
      lo,
      hi,
      linesOverlap,
      q25,
      q75,
      percentileWindow,
    }
  }, [data])

  const barClass = (tone: 'high' | 'low' | 'mid') => {
    if (tone === 'high') return 'bg-gradient-to-t from-amber-600 to-amber-400'
    if (tone === 'low') return 'bg-gradient-to-t from-slate-500 to-slate-400'
    return 'bg-gradient-to-t from-slate-400 to-slate-300'
  }

  const pw = data.zoom?.percentile_window

  return (
    <div className="w-full min-w-0">
      <p className="font-mono text-[9px] text-slate-600 mb-1 font-semibold tracking-wide">{title}</p>
      {pw ? (
        <p className="text-[8px] text-slate-500 mb-1.5 leading-snug">
          View: p{pw[0]}–p{pw[1]} range ({lo.toFixed(2)} to {hi.toFixed(2)}) — full data span{' '}
          {data.full_range?.min != null && data.full_range?.max != null
            ? `[${data.full_range.min.toFixed(2)}, ${data.full_range.max.toFixed(2)}]`
            : ''}
          . Bar height uses √count so smaller bins stay visible.
        </p>
      ) : null}

      <div className="rounded-lg border border-slate-200 bg-white shadow-inner overflow-hidden">
        <div className="relative h-[120px] border-b border-slate-100">
          {/* light grid */}
          {[0.25, 0.5, 0.75].map((t) => (
            <div
              key={t}
              className="absolute left-0 right-0 border-t border-slate-100 pointer-events-none"
              style={{ bottom: `${t * 100}%` }}
            />
          ))}
          <div className="absolute inset-0 flex items-end gap-[2px] px-1.5 pb-0 pt-2">
            {bars.map((b, i) => (
              <div
                key={i}
                className={`min-w-0 flex-1 rounded-t-[2px] min-h-[2px] shadow-sm ${barClass(b.tone)}`}
                style={{ height: `${Math.max(4, b.h)}%` }}
                title={`count=${b.c}`}
              />
            ))}
          </div>
          <div
            className="absolute top-2 bottom-0 w-0 border-l-2 border-dashed border-teal-600 z-10 pointer-events-none opacity-90"
            style={{ left: `calc(${p25}% )` }}
            title={`${labelQ25} (${q25.toFixed(3)})`}
          />
          <div
            className="absolute top-2 bottom-0 w-0 border-l-2 border-dashed border-amber-600 z-10 pointer-events-none opacity-90"
            style={{ left: `calc(${p75}% )` }}
            title={`${labelQ75} (${q75.toFixed(3)})`}
          />
        </div>
        <div className="flex justify-between items-start gap-2 px-2 py-1.5 bg-slate-50/80">
          <span className="text-[7px] font-mono text-slate-500 tabular-nums">{lo.toFixed(3)}</span>
          {linesOverlap ? (
            <span className="text-[7px] font-mono text-slate-600 text-center flex-1">
              Q25 ≈ Q75 at display scale (IQR tiny: {(q75 - q25).toExponential(2)})
            </span>
          ) : (
            <span className="text-[7px] font-mono text-slate-600 text-center flex-1">
              <span className="text-teal-700">Q25</span>={q25.toFixed(3)} ·{' '}
              <span className="text-amber-800">Q75</span>={q75.toFixed(3)}
            </span>
          )}
          <span className="text-[7px] font-mono text-slate-500 tabular-nums">{hi.toFixed(3)}</span>
        </div>
      </div>
      <p className="text-[8px] text-slate-500 mt-1.5 text-center font-mono">{axisLabel}</p>
      <p className="text-[9px] text-slate-600 mt-1.5 leading-snug">{caption}</p>
    </div>
  )
}
