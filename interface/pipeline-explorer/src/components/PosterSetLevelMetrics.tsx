/** Radar summary for set-level metrics + stability note (poster). */

type Labels = {
  radarHelp: string
  stabilityHeading: string
  stabilityNoData: string
  axisCov: string
  axisConc: string
  axisAuc: string
  axisF1: string
  axisDistinct: string
  legendOur: string
  legendRef: string
  tableTitle: string
}

type Props = {
  coverage: number | undefined
  concentration: number | undefined
  auc: number | undefined
  f1: number | undefined
  avgJaccard: number | undefined
  stabilityDetail: string | null
  labels: Labels
}

const CX = 80
const CY = 80
const R_MAX = 56
const N = 5

function ringPointsFixed(values: (number | undefined)[]): string {
  const pts: string[] = []
  for (let i = 0; i < N; i++) {
    const v = Math.max(0, Math.min(1, values[i] ?? 0))
    const angleDeg = -90 + (360 / N) * i
    const rad = (angleDeg * Math.PI) / 180
    const x = CX + R_MAX * v * Math.cos(rad)
    const y = CY + R_MAX * v * Math.sin(rad)
    pts.push(`${x},${y}`)
  }
  return pts.join(' ')
}

function axisLabelPos(i: number): { x: number; y: number; anchor: 'start' | 'middle' | 'end' } {
  const angleDeg = -90 + (360 / N) * i
  const rad = (angleDeg * Math.PI) / 180
  const lr = R_MAX + 14
  const x = CX + lr * Math.cos(rad)
  const y = CY + lr * Math.sin(rad)
  let anchor: 'start' | 'middle' | 'end' = 'middle'
  if (Math.cos(rad) < -0.3) anchor = 'end'
  if (Math.cos(rad) > 0.3) anchor = 'start'
  return { x, y, anchor }
}

export function PosterSetLevelMetrics({
  coverage,
  concentration,
  auc,
  f1,
  avgJaccard,
  stabilityDetail,
  labels: L,
}: Props) {
  const distinct = avgJaccard != null && Number.isFinite(avgJaccard) ? 1 - avgJaccard : undefined
  const vals = [coverage, concentration, auc, f1, distinct]
  const baseline = [0.5, 0.5, 0.5, 0.5, 0.5]

  const axisLabels = [L.axisCov, L.axisConc, L.axisAuc, L.axisF1, L.axisDistinct]

  return (
    <div className="w-full min-w-0">
      <div className="grid grid-cols-1 sm:grid-cols-[160px_1fr] gap-2 items-start">
        <svg viewBox="0 0 160 160" className="w-full max-w-[160px] mx-auto sm:mx-0" aria-hidden>
          {/* pentagon rings */}
          {[0.25, 0.5, 0.75, 1].map((t) => (
            <polygon
              key={t}
              fill="none"
              stroke="#cbd5e1"
              strokeWidth={t === 0.5 ? 1.2 : 0.6}
              strokeDasharray={t === 0.5 ? '4 3' : undefined}
              opacity={t === 0.5 ? 0.9 : 0.45}
              points={ringPointsFixed([t, t, t, t, t])}
            />
          ))}
          {/* Reference pentagon @ 0.5 — not a tuned baseline, visual guide only */}
          <polygon
            fill="rgba(245, 158, 11, 0.08)"
            stroke="#d97706"
            strokeWidth={1.4}
            strokeDasharray="5 4"
            points={ringPointsFixed(baseline)}
          />
          <polygon
            fill="rgba(13, 148, 136, 0.2)"
            stroke="#0f766e"
            strokeWidth={1.75}
            points={ringPointsFixed(vals)}
          />
          {axisLabels.map((lab, i) => {
            const pos = axisLabelPos(i)
            return (
              <text
                key={lab}
                x={pos.x}
                y={pos.y}
                textAnchor={pos.anchor}
                className="fill-slate-600"
                style={{ fontSize: '7px', fontFamily: 'ui-monospace, monospace' }}
              >
                {lab}
              </text>
            )
          })}
        </svg>
        <div className="text-[9px] text-slate-600 space-y-1.5">
          <p className="font-mono text-[8px] uppercase tracking-wide text-teal-800">{L.tableTitle}</p>
          <ul className="space-y-0.5 tabular-nums">
            <li>
              <span className="text-slate-500">{L.axisCov}:</span>{' '}
              <strong className="text-slate-900">{coverage != null ? coverage.toFixed(3) : '—'}</strong>
            </li>
            <li>
              <span className="text-slate-500">{L.axisConc}:</span>{' '}
              <strong className="text-slate-900">{concentration != null ? concentration.toFixed(3) : '—'}</strong>
            </li>
            <li>
              <span className="text-slate-500">{L.axisAuc}:</span>{' '}
              <strong className="text-slate-900">{auc != null ? auc.toFixed(3) : '—'}</strong>
            </li>
            <li>
              <span className="text-slate-500">{L.axisF1}:</span>{' '}
              <strong className="text-slate-900">{f1 != null ? f1.toFixed(3) : '—'}</strong>
            </li>
            <li>
              <span className="text-slate-500">{L.axisDistinct}:</span>{' '}
              <strong className="text-slate-900">{distinct != null ? distinct.toFixed(3) : '—'}</strong>{' '}
              <span className="text-slate-400">(1 − Jaccard)</span>
            </li>
          </ul>
          <p className="text-[8.5px] text-slate-500 border-l-2 border-teal-200 pl-2 leading-snug mt-1">{L.radarHelp}</p>
          <p className="text-[8px] text-slate-500 pt-1">
            <span className="inline-block w-2 h-2 rounded-sm bg-teal-500/30 border border-teal-600 mr-1 align-middle" />
            {L.legendOur}{' '}
            <span className="inline-block w-2 h-2 rounded-sm bg-amber-100 border border-amber-600 border-dashed mr-1 align-middle ml-2" />
            {L.legendRef}
          </p>
        </div>
      </div>

      <div className="mt-3 pt-2 border-t border-slate-200">
        <div className="flex items-center gap-2 mb-1.5">
          <div className="flex-1 h-px bg-slate-200" />
          <span className="font-mono text-[8px] uppercase tracking-widest text-slate-500">{L.stabilityHeading}</span>
          <div className="flex-1 h-px bg-slate-200" />
        </div>
        {stabilityDetail ? (
          <p className="text-[9px] text-slate-600 leading-relaxed">{stabilityDetail}</p>
        ) : (
          <p className="text-[9px] text-slate-600 leading-relaxed">{L.stabilityNoData}</p>
        )}
      </div>
    </div>
  )
}
