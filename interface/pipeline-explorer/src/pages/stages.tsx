import { useEffect, useState } from 'react'
import { LoadState } from '../components/LoadState'
import { MaterialIcon } from '../components/MaterialIcon'
import { PipelineStepper } from '../components/PipelineStepper'
import { P } from '../paths'
import type {
  DiffDiagnostics,
  EvalMetrics,
  ExportMeta,
  PatternCatalog,
  ReleaseReadiness,
  ResidualProfile,
  SaeMeta,
} from '../types'
import { fetchText, useFetchJson } from '../useFetchJson'

function StageHeader({
  step,
  accent,
  title,
  description,
}: {
  step: string
  accent: string
  title: string
  description: string
}) {
  return (
    <div className="flex justify-between items-end flex-wrap gap-4 mb-8">
      <div>
        <span
          className="label-sm font-bold uppercase tracking-widest mb-2 block text-sm"
          style={{ color: accent }}
        >
          {step}
        </span>
        <h2 className="text-3xl font-extrabold text-on-surface font-headline leading-tight">{title}</h2>
        <p className="text-on-surface-variant mt-2 max-w-2xl font-body">{description}</p>
      </div>
    </div>
  )
}

export function ResidualStage() {
  const { data, error, loading } = useFetchJson<ResidualProfile>(P.residualProfile)
  const meta = useFetchJson<{ group_counts?: Record<string, number>; n_rows?: number }>(P.residualMeta)

  const subjects = [...(data?.by_subject ?? [])].sort((a, b) => b.error_rate - a.error_rate).slice(0, 15)
  const groups = data?.group_counts ?? meta.data?.group_counts
  const maxG = groups ? Math.max(...Object.values(groups), 1) : 1

  return (
    <>
      <PipelineStepper activeId="residual" />
      <StageHeader
        step="Stage 01: Residuals"
        accent="#3B82F6"
        title="Residual analysis — error by subject"
        description="Profile over the full MMLU residual export (GPT-3.5 predictions): error rates and residual groups from mmlu_gpt35_residuals_profile.json."
      />
      <LoadState loading={loading || meta.loading} error={error ?? meta.error}>
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-7 bg-surface-container-lowest rounded-xl p-6">
            <h3 className="font-headline font-bold mb-2">Error rate by subject</h3>
            <p className="text-sm text-on-surface-variant mb-4">mmlu_gpt35_residuals_error_by_subject.png</p>
            <img
              src={P.residualPng}
              alt="Error by subject"
              className="w-full rounded-lg bg-surface-container-low"
            />
          </div>
          <div className="col-span-12 lg:col-span-5 space-y-6">
            <div className="bg-surface-container-low rounded-xl p-6">
              <h3 className="font-headline font-bold mb-4">Cohort</h3>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-on-surface-variant">Rows</dt>
                  <dd className="font-semibold">{data?.n_rows?.toLocaleString() ?? '—'}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-on-surface-variant">Error rate</dt>
                  <dd className="font-semibold">
                    {data?.error_rate != null ? `${(data.error_rate * 100).toFixed(2)}%` : '—'}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-on-surface-variant">Expected error (mean)</dt>
                  <dd className="font-semibold">{data?.expected_error_mean?.toFixed(4) ?? '—'}</dd>
                </div>
              </dl>
            </div>
            {groups ? (
              <div className="bg-surface-container-lowest rounded-xl p-6">
                <h3 className="font-headline font-bold mb-4">Residual groups</h3>
                <div className="space-y-3">
                  {Object.entries(groups).map(([k, v]) => (
                    <div key={k}>
                      <div className="flex justify-between text-xs mb-1">
                        <span>{k}</span>
                        <span className="font-semibold">{v.toLocaleString()}</span>
                      </div>
                      <div className="h-2 rounded-full bg-surface-container-low overflow-hidden">
                        <div
                          className="h-full rounded-full bg-[#3B82F6]"
                          style={{ width: `${(v / maxG) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          <div className="col-span-12 bg-surface-container-lowest rounded-xl overflow-hidden">
            <div className="px-6 py-3 bg-surface-container-high font-semibold text-sm">Highest error-rate subjects (top 15)</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-on-surface-variant border-b border-outline-variant/15">
                    <th className="p-3 pl-6">Subject</th>
                    <th className="p-3">n</th>
                    <th className="p-3">Errors</th>
                    <th className="p-3 pr-6">Error rate</th>
                  </tr>
                </thead>
                <tbody>
                  {subjects.map((row, i) => (
                    <tr
                      key={row.subject}
                      className={i % 2 === 0 ? 'bg-surface-container-low/40' : 'bg-surface-container-lowest'}
                    >
                      <td className="p-3 pl-6 border-l-4 border-[#3B82F6]">{row.subject}</td>
                      <td className="p-3">{row.n}</td>
                      <td className="p-3">{row.errors}</td>
                      <td className="p-3 pr-6">{(row.error_rate * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </LoadState>
    </>
  )
}

export function ExportStage() {
  const { data, error, loading } = useFetchJson<ExportMeta>(P.exportMeta)
  return (
    <>
      <PipelineStepper activeId="export" />
      <StageHeader
        step="Stage 02: Export"
        accent="#6366F1"
        title="HF hidden activations"
        description="Pooled last-token hidden states written to NPZ; metadata from mmlu_hf_hidden.meta.json."
      />
      <LoadState loading={loading} error={error}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-surface-container-lowest rounded-xl p-8 space-y-4">
            <h3 className="font-headline font-bold">Model & tensor</h3>
            <dl className="space-y-3 text-sm">
              <Row label="Model" value={data?.model} />
              <Row label="Rows" value={data?.n_rows?.toLocaleString()} />
              <Row label="Hidden dim" value={data?.hidden_dim?.toString()} />
              <Row label="Layer index" value={data?.layer_index?.toString()} />
            </dl>
          </div>
          <div className="bg-surface-container-low rounded-xl p-8 flex flex-col justify-center items-center text-center">
            <MaterialIcon name="dataset" className="text-5xl text-secondary mb-4" />
            <p className="text-on-surface-variant text-sm max-w-sm">
              Binary latent tensor: <code className="text-xs bg-surface-container-lowest px-1 rounded">mmlu_hf_hidden.npz</code>{' '}
              (served under /pipeline_2026 for download or external tools).
            </p>
          </div>
        </div>
      </LoadState>
    </>
  )
}

function Row({ label, value }: { label: string; value?: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-outline-variant/10 pb-2">
      <dt className="text-on-surface-variant">{label}</dt>
      <dd className="font-semibold text-right">{value ?? '—'}</dd>
    </div>
  )
}

export function SaeStage() {
  const { data, error, loading } = useFetchJson<SaeMeta>(P.saeMeta)
  return (
    <>
      <PipelineStepper activeId="sae" />
      <StageHeader
        step="Stage 03: SAE encode"
        accent="#8B5CF6"
        title="Sparse autoencoder latents"
        description="SAELens encoding of exported hiddens: mmlu_hf_sae_latents.meta.json."
      />
      <LoadState loading={loading} error={error}>
        <div className="bg-surface-container-lowest rounded-xl p-8 max-w-3xl">
          <dl className="space-y-3 text-sm">
            <Row label="Combined model string" value={data?.model} />
            <Row label="Rows" value={data?.n_rows?.toLocaleString()} />
            <Row label="d_sae" value={data?.d_sae?.toLocaleString()} />
            <Row label="SAE release" value={data?.sae_release} />
            <Row label="SAE hook" value={data?.sae_id} />
          </dl>
          <p className="text-xs text-on-surface-variant mt-6">
            Output: <code className="bg-surface-container-low px-1 rounded">mmlu_hf_sae_latents.npz</code>
          </p>
        </div>
      </LoadState>
    </>
  )
}

export function DiffStage() {
  const diag = useFetchJson<DiffDiagnostics>(P.diffDiagnostics)
  const catalog = useFetchJson<PatternCatalog>(P.patternCatalog)
  const rows = (catalog.data?.patterns ?? []).slice(0, 25)

  return (
    <>
      <PipelineStepper activeId="diff" />
      <StageHeader
        step="Stage 04: Diff"
        accent="#EC4899"
        title="Latent diff & patterns"
        description="Group-wise tests over SAE latents: diagnostics and top patterns from pattern_catalog.json."
      />
      <LoadState loading={diag.loading || catalog.loading} error={diag.error ?? catalog.error}>
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 md:col-span-4 space-y-4">
            {[
              ['High residual n', diag.data?.n_high_residual],
              ['Low residual n', diag.data?.n_low_residual],
              ['Latents tested', diag.data?.n_latents_tested?.toLocaleString()],
              ['Passing patterns', diag.data?.n_passing?.toLocaleString()],
            ].map(([k, v]) => (
              <div key={String(k)} className="bg-surface-container-highest rounded-lg p-4">
                <p className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant/60">{k}</p>
                <p className="font-headline text-2xl font-bold">{v ?? '—'}</p>
              </div>
            ))}
          </div>
          <div className="col-span-12 md:col-span-8 bg-surface-container-lowest rounded-xl overflow-hidden">
            <div className="px-6 py-3 bg-surface-container-high font-semibold text-sm">
              Top patterns (by catalog order) · membership q={catalog.data?.membership_quantile ?? '—'}
            </div>
            <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface-container-high">
                  <tr className="text-left text-on-surface-variant">
                    <th className="p-2 pl-4">pattern_id</th>
                    <th className="p-2">latent_id</th>
                    <th className="p-2">V′</th>
                    <th className="p-2">p</th>
                    <th className="p-2 pr-4">threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={r.pattern_id} className={i % 2 === 0 ? 'bg-surface-container-low/30' : ''}>
                      <td className="p-2 pl-4 border-l-4 border-[#EC4899]">{r.pattern_id}</td>
                      <td className="p-2">{r.latent_id}</td>
                      <td className="p-2">{r.V_prime?.toFixed(4)}</td>
                      <td className="p-2">{r.p_value?.toExponential(3)}</td>
                      <td className="p-2 pr-4">{r.threshold?.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </LoadState>
    </>
  )
}

type LeaderRow = Record<string, string>

function parseLeaderboardCsv(text: string): LeaderRow[] {
  const lines = text.trim().split('\n')
  if (lines.length < 2) return []
  const headers = lines[0].split(',')
  return lines.slice(1).map((line) => {
    const cells = line.split(',')
    const row: LeaderRow = {}
    headers.forEach((h, i) => {
      row[h.trim()] = (cells[i] ?? '').trim()
    })
    return row
  })
}

export function EvalStage() {
  const { data, error, loading } = useFetchJson<EvalMetrics>(P.evalMetrics)
  const [lb, setLb] = useState<{ rows: LeaderRow[]; err: Error | null }>({ rows: [], err: null })

  useEffect(() => {
    fetchText(P.leaderboard)
      .then((t) => setLb({ rows: parseLeaderboardCsv(t), err: null }))
      .catch((e: unknown) =>
        setLb({ rows: [], err: e instanceof Error ? e : new Error(String(e)) }),
      )
  }, [])

  const rm = data?.residual_metrics
  const sl = data?.set_level_metrics

  return (
    <>
      <PipelineStepper activeId="eval" />
      <StageHeader
        step="Stage 05: Eval"
        accent="#10B981"
        title="Metrics & leaderboard"
        description="Head-200 eval summary and leaderboard.csv slice for this run."
      />
      <LoadState loading={loading} error={error}>
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-6 bg-surface-container-lowest rounded-xl p-6">
            <h3 className="font-headline font-bold mb-2">Residual stratification</h3>
            <img
              src={P.evalResidualBar}
              alt="Residual bar"
              className="w-full rounded-lg bg-surface-container-low mt-4"
            />
          </div>
          <div className="col-span-12 lg:col-span-6 grid grid-cols-2 gap-4">
            <div className="bg-surface-container-highest rounded-lg p-4">
              <p className="text-[10px] font-bold uppercase text-on-surface-variant/60">High residual coverage</p>
              <p className="text-2xl font-bold font-headline">
                {rm?.high_residual?.coverage_of_errors != null
                  ? `${(rm.high_residual.coverage_of_errors * 100).toFixed(1)}%`
                  : '—'}
              </p>
              <p className="text-xs text-on-surface-variant mt-1">of errors</p>
            </div>
            <div className="bg-surface-container-highest rounded-lg p-4">
              <p className="text-[10px] font-bold uppercase text-on-surface-variant/60">Coverage / concentration</p>
              <p className="text-2xl font-bold font-headline">
                {sl?.coverage_concentration?.coverage != null
                  ? `${(sl.coverage_concentration.coverage * 100).toFixed(0)}%`
                  : '—'}{' '}
                <span className="text-base font-normal text-on-surface-variant">/</span>{' '}
                {sl?.coverage_concentration?.concentration != null
                  ? sl.coverage_concentration.concentration.toFixed(3)
                  : '—'}
              </p>
            </div>
            <div className="bg-surface-container-highest rounded-lg p-4 col-span-2">
              <p className="text-[10px] font-bold uppercase text-on-surface-variant/60">Predictive utility</p>
              <p className="text-xl font-bold">
                AUC {sl?.predictive_utility?.auc?.toFixed(4) ?? '—'} · F1{' '}
                {sl?.predictive_utility?.f1?.toFixed(4) ?? '—'}
              </p>
            </div>
          </div>
          <div className="col-span-12 bg-surface-container-lowest rounded-xl overflow-hidden">
            <div className="px-6 py-3 bg-surface-container-high font-semibold text-sm flex justify-between">
              <span>leaderboard.csv</span>
              {lb.err ? <span className="text-error text-xs font-normal">{lb.err.message}</span> : null}
            </div>
            <div className="overflow-x-auto">
              {lb.rows.length === 0 && !lb.err ? (
                <p className="p-6 text-sm text-on-surface-variant">No rows parsed.</p>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-on-surface-variant border-b border-outline-variant/15">
                      {lb.rows[0]
                        ? Object.keys(lb.rows[0]).map((h) => (
                            <th key={h} className="p-2 px-3">
                              {h}
                            </th>
                          ))
                        : null}
                    </tr>
                  </thead>
                  <tbody>
                    {lb.rows.map((row, i) => (
                      <tr
                        key={i}
                        className={`border-l-4 border-[#10B981] ${i % 2 === 0 ? 'bg-surface-container-low/40' : ''}`}
                      >
                        {lb.rows[0]
                          ? Object.keys(lb.rows[0]).map((k) => (
                              <td key={k} className="p-2 px-3">
                                {row[k] ?? ''}
                              </td>
                            ))
                          : null}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      </LoadState>
    </>
  )
}

export function ReportStage() {
  const { data, error, loading } = useFetchJson<ReleaseReadiness>(P.releaseReadiness)
  const reportUrl = P.reportHtml

  return (
    <>
      <PipelineStepper activeId="report" />
      <StageHeader
        step="Stage 06: Report"
        accent="#0058BE"
        title="Release readiness & HTML report"
        description="Aggregated gate verdict from release_readiness.json; full narrative report is static HTML."
      />
      <LoadState loading={loading} error={error}>
        <div className="space-y-6">
          <a
            href={reportUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-white bg-gradient-to-r from-primary to-primary-container"
          >
            <MaterialIcon name="open_in_new" />
            Open mmlu_report.html
          </a>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-surface-container-lowest rounded-xl p-6 min-h-[360px]">
              <h3 className="font-headline font-bold mb-4">Embedded preview</h3>
              <iframe title="MMLU report" src={reportUrl} className="w-full h-[480px] rounded-lg border border-outline-variant/15 bg-white" />
            </div>
            <div className="bg-surface-container-low rounded-xl p-6 text-sm space-y-4">
              <h3 className="font-headline font-bold">JSON highlights</h3>
              <pre className="text-xs overflow-auto max-h-[480px] bg-surface-container-lowest p-4 rounded-lg text-on-surface-variant">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      </LoadState>
    </>
  )
}
