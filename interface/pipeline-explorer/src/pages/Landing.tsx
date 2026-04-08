import { Link } from 'react-router-dom'
import { MaterialIcon } from '../components/MaterialIcon'
import { PipelineStepper } from '../components/PipelineStepper'
import { LoadState } from '../components/LoadState'
import { P, fileNameOnly } from '../paths'
import { useFetchJson } from '../useFetchJson'
import type { EvalMetrics, PaperMmluSummary, ReleaseReadiness } from '../types'

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="bg-surface-container-highest rounded-lg p-4">
      <p className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant/60 mb-1">
        {label}
      </p>
      <p className="font-headline text-2xl font-bold text-on-surface">{value}</p>
      {hint ? <p className="text-xs text-on-surface-variant mt-1">{hint}</p> : null}
    </div>
  )
}

export function Landing() {
  const rr = useFetchJson<ReleaseReadiness>(P.releaseReadiness)
  const paper = useFetchJson<PaperMmluSummary>(P.paperMmluSummary)
  const ev = useFetchJson<EvalMetrics>(P.evalMetrics)

  const loading = rr.loading || paper.loading || ev.loading
  const error = rr.error ?? paper.error ?? ev.error

  const runLabel =
    ev.data?.label ?? 'mmlu_gpt35_sae_latents_head_200'
  const behavioral = rr.data?.behavioral_model?.mode ?? 'ChatGPT (residual export)'
  const repr = rr.data?.representation_model?.model ?? '—'

  const verdict = rr.data?.gates?.verdict ?? 'unknown'
  const pass = verdict === 'pass'
  const overall = rr.data?.overall

  const fileHints = [
    rr.data?.representation_model?.residual_csv,
    rr.data?.representation_model?.source_hidden_meta,
    rr.data?.representation_model?.output_npz,
    rr.data?.final_set_level_metrics?.pattern_catalog_source,
  ].filter(Boolean) as string[]

  return (
    <div className="min-h-screen bg-surface text-on-surface font-body">
      <header className="bg-surface fixed top-0 left-0 w-full z-50 flex justify-between items-center px-8 py-3">
        <div className="flex items-center gap-8">
          <span className="text-xl font-bold tracking-tight font-headline">Pipeline Explorer</span>
          <nav className="hidden md:flex gap-6 text-sm">
            <span className="text-primary font-semibold border-b-2 border-primary py-1">Run selection</span>
            <Link className="text-on-surface/60 hover:text-on-surface py-1" to="/stage/residual">
              Explorer
            </Link>
          </nav>
        </div>
        <Link
          to="/stage/residual"
          className="bg-gradient-to-r from-primary to-primary-container text-white px-6 py-2 rounded-xl font-bold text-sm flex items-center gap-2"
        >
          <MaterialIcon name="hub" />
          Open stage explorer
        </Link>
      </header>

      <main className="pt-24 px-8 max-w-[1400px] mx-auto pb-20">
        <section className="mb-12">
          <div className="mb-2">
            <span className="font-body text-xs font-bold tracking-[0.2em] text-primary uppercase">
              Current configuration
            </span>
          </div>
          <div className="flex flex-col md:flex-row justify-between items-end gap-6">
            <div className="max-w-2xl">
              <h1 className="font-headline text-4xl md:text-5xl font-extrabold tracking-tight mb-4 leading-tight">
                {runLabel}
              </h1>
              <p className="text-lg text-on-surface-variant leading-relaxed">
                Live view of <code className="text-sm bg-surface-container-low px-1.5 py-0.5 rounded">pipeline_2026</code>{' '}
                outputs: MMLU residual analysis, HF hidden export, SAE latents, diff mining, eval metrics, and release
                readiness. Stage 1 paper figures and summary are included for context.
              </p>
            </div>
            <div className="flex flex-col items-end gap-2 text-xs text-on-surface-variant/70">
              {paper.data?.n_rows != null ? (
                <span>Paper notebook cohort: {paper.data.n_rows.toLocaleString()} rows</span>
              ) : null}
              {paper.data?.error_rate != null ? (
                <span>Paper cohort error rate: {(paper.data.error_rate * 100).toFixed(2)}%</span>
              ) : null}
            </div>
          </div>
        </section>

        <LoadState loading={loading} error={error}>
          <PipelineStepper />

          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-8 bg-surface-container-lowest p-8 rounded-2xl shadow-float">
              <div className="flex justify-between items-start mb-8">
                <div>
                  <h2 className="font-headline text-2xl font-bold mb-1">Manifest details</h2>
                  <p className="text-sm text-on-surface-variant">Provenance from release_readiness.json and stage metas.</p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                <div className="space-y-6">
                  <div>
                    <span className="font-body text-[10px] font-black tracking-widest text-on-surface-variant/60 uppercase block mb-2">
                      Behavioral model
                    </span>
                    <div className="bg-surface-container-low p-4 rounded-xl flex items-center gap-3">
                      <MaterialIcon name="psychology" className="text-primary text-2xl" />
                      <div>
                        <p className="font-bold text-on-surface">{behavioral}</p>
                        <p className="text-xs text-on-surface-variant">Residual CSV column / export label</p>
                      </div>
                    </div>
                  </div>
                  <div>
                    <span className="font-body text-[10px] font-black tracking-widest text-on-surface-variant/60 uppercase block mb-2">
                      Representation model
                    </span>
                    <div className="bg-surface-container-low p-4 rounded-xl flex items-center gap-3">
                      <MaterialIcon name="neurology" className="text-secondary text-2xl" />
                      <div>
                        <p className="font-bold text-on-surface text-sm leading-snug">{repr}</p>
                        <p className="text-xs text-on-surface-variant mt-1">
                          {rr.data?.representation_model?.sae_release ?? '—'} ·{' '}
                          {rr.data?.representation_model?.sae_id ?? '—'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                <div>
                  <span className="font-body text-[10px] font-black tracking-widest text-on-surface-variant/60 uppercase block mb-2">
                    Key artifact paths
                  </span>
                  <div className="space-y-2">
                    {fileHints.slice(0, 6).map((f) => (
                      <div
                        key={f}
                        className="flex items-center justify-between p-3 rounded-lg hover:bg-surface-container-low transition-colors"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <MaterialIcon name="description" className="text-on-surface-variant/40 shrink-0" />
                          <span className="text-sm font-medium truncate" title={f}>
                            {fileNameOnly(f)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {overall ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-10">
                  <MetricCard
                    label="Eval instances"
                    value={String(overall.n_instances ?? '—')}
                  />
                  <MetricCard
                    label="Errors"
                    value={String(overall.n_errors ?? '—')}
                  />
                  <MetricCard
                    label="Error rate"
                    value={
                      overall.error_rate != null ? `${(overall.error_rate * 100).toFixed(1)}%` : '—'
                    }
                  />
                  <MetricCard
                    label="Accuracy"
                    value={
                      overall.accuracy != null ? `${(overall.accuracy * 100).toFixed(1)}%` : '—'
                    }
                  />
                </div>
              ) : null}
            </div>

            <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
              <div className="bg-surface-container-highest p-8 rounded-2xl flex-1 relative overflow-hidden">
                <div className="absolute -right-10 -bottom-10 opacity-10 pointer-events-none">
                  <MaterialIcon name="verified" className="text-[160px]" />
                </div>
                <div className="relative z-10">
                  <span className="font-body text-[10px] font-black tracking-widest text-on-surface-variant/60 uppercase block mb-4">
                    Release readiness
                  </span>
                  <div className="flex items-center gap-4 mb-6">
                    <div
                      className={`w-16 h-16 rounded-full flex items-center justify-center text-white shadow-lg ${
                        pass ? 'bg-[#10B981] shadow-[#10B981]/20' : 'bg-error shadow-error/20'
                      }`}
                    >
                      <MaterialIcon name={pass ? 'check_circle' : 'cancel'} className="text-3xl" />
                    </div>
                    <div>
                      <h4 className="font-headline text-3xl font-black text-on-surface uppercase">
                        {pass ? 'Go' : 'No-go'}
                      </h4>
                      <p className={`text-sm font-semibold ${pass ? 'text-[#10B981]' : 'text-error'}`}>
                        Verdict: {verdict}
                      </p>
                    </div>
                  </div>
                  {rr.data?.gates?.violations?.length ? (
                    <ul className="text-sm text-on-surface-variant space-y-2 mb-6 list-disc pl-4">
                      {rr.data.gates.violations.map((v, i) => (
                        <li key={i}>
                          <span className="font-semibold text-on-surface">{v.gate}</span>: {v.detail}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-on-surface-variant text-sm leading-relaxed mb-6">
                      Gates evaluated against thresholds in release_readiness.json.
                    </p>
                  )}
                  <Link
                    to="/stage/report"
                    className="block w-full bg-surface-container-lowest py-3 rounded-xl font-bold text-on-surface text-center shadow-sm hover:-translate-y-0.5 transition-transform"
                  >
                    Review report stage
                  </Link>
                </div>
              </div>

              {ev.data?.set_level_metrics?.predictive_utility ? (
                <div className="bg-surface-container-low p-6 rounded-2xl">
                  <p className="text-[10px] font-black tracking-widest text-on-surface-variant/60 uppercase mb-2">
                    Predictive utility (head-200)
                  </p>
                  <p className="font-headline text-2xl font-bold">
                    AUC {ev.data.set_level_metrics.predictive_utility.auc?.toFixed(3) ?? '—'}
                  </p>
                  <p className="text-sm text-on-surface-variant mt-1">
                    F1 {ev.data.set_level_metrics.predictive_utility.f1?.toFixed(3) ?? '—'} · Features{' '}
                    {ev.data.set_level_metrics.predictive_utility.n_features ?? '—'}
                  </p>
                </div>
              ) : null}
            </div>
          </div>

          {paper.data?.figures_written?.length ? (
            <section className="mt-12">
              <h3 className="font-headline text-lg font-bold mb-4">Stage 1 figures (paper_mmlu_summary)</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {paper.data.figures_written.slice(0, 8).map((fig) => (
                  <a
                    key={fig}
                    href={`/pipeline_2026/stage1/results/${fig}`}
                    target="_blank"
                    rel="noreferrer"
                    className="block bg-surface-container-lowest rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow border-l-4 border-primary"
                  >
                    <img
                      src={`/pipeline_2026/stage1/results/${fig}`}
                      alt={fig}
                      className="w-full h-32 object-cover object-top bg-surface-container-low"
                    />
                    <p className="text-xs p-2 truncate font-medium" title={fig}>
                      {fig}
                    </p>
                  </a>
                ))}
              </div>
            </section>
          ) : null}
        </LoadState>
      </main>
    </div>
  )
}
