import { useMemo, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { MaterialIcon } from '../components/MaterialIcon'
import { LoadState } from '../components/LoadState'
import { P } from '../paths'
import { useFetchJson } from '../useFetchJson'
import type { DiffDiagnostics, EvalMetrics, ReleaseReadiness } from '../types'

type Mode = 'beginner' | 'technical'
type StageId = 'data' | 'residual' | 'sae' | 'diff' | 'eval'

function ToggleMode({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <div className="rounded-xl border border-outline-variant/30 p-1 flex items-center gap-1">
      <button
        type="button"
        onClick={() => onChange('beginner')}
        className={`px-3 py-1 text-xs rounded-lg font-semibold ${mode === 'beginner' ? 'bg-primary text-white' : 'text-on-surface-variant'}`}
      >
        Beginner Mode
      </button>
      <button
        type="button"
        onClick={() => onChange('technical')}
        className={`px-3 py-1 text-xs rounded-lg font-semibold ${mode === 'technical' ? 'bg-primary text-white' : 'text-on-surface-variant'}`}
      >
        Technical Mode
      </button>
    </div>
  )
}

function ConceptTooltip({ term, body }: { term: string; body: string }) {
  return (
    <span className="relative inline-flex group">
      <span className="underline decoration-dotted cursor-help">{term}</span>
      <span
        role="tooltip"
        className="pointer-events-none absolute z-20 left-1/2 -translate-x-1/2 top-7 w-72 rounded-md bg-surface-container-highest text-on-surface text-xs leading-snug p-2 shadow-lg border border-outline-variant/20 opacity-0 translate-y-1 transition-all duration-150 group-hover:opacity-100 group-hover:translate-y-0 group-focus-within:opacity-100 group-focus-within:translate-y-0"
      >
        {body}
      </span>
    </span>
  )
}

function ExampleBox({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="bg-surface-container-low rounded-xl p-4">
      <p className="text-sm font-semibold mb-2">{title}</p>
      <pre className="text-xs whitespace-pre-wrap text-on-surface-variant">{children}</pre>
    </div>
  )
}

function MetricTable({
  auc,
  f1,
  jaccard,
}: {
  auc?: number
  f1?: number
  jaccard?: number
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-outline-variant/20">
      <table className="w-full text-sm">
        <thead className="bg-surface-container-high">
          <tr className="text-left">
            <th className="p-3">Metric</th>
            <th className="p-3">Value</th>
            <th className="p-3">Meaning</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-t border-outline-variant/10">
            <td className="p-3">AUC</td>
            <td className="p-3">{auc?.toFixed(3) ?? '—'}</td>
            <td className="p-3">Model cannot rank failures reliably</td>
          </tr>
          <tr className="border-t border-outline-variant/10">
            <td className="p-3">F1</td>
            <td className="p-3">{f1?.toFixed(3) ?? '—'}</td>
            <td className="p-3">Moderate classification quality</td>
          </tr>
          <tr className="border-t border-outline-variant/10">
            <td className="p-3">Jaccard</td>
            <td className="p-3">{jaccard?.toFixed(3) ?? '—'}</td>
            <td className="p-3">High overlap indicates redundancy</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function PipelineDiagram({
  selected,
  onSelect,
}: {
  selected: StageId
  onSelect: (stage: StageId) => void
}) {
  const items: { id: StageId; label: string; hint: string }[] = [
    { id: 'data', label: 'Data', hint: 'Input questions and model predictions.' },
    { id: 'residual', label: 'Residual', hint: 'Control question difficulty.' },
    { id: 'sae', label: 'SAE Encode', hint: 'Transform hidden states to sparse features.' },
    { id: 'diff', label: 'Diff', hint: 'Extract patterns that separate groups.' },
    { id: 'eval', label: 'Eval', hint: 'Check if patterns are useful.' },
  ]

  return (
    <div className="flex flex-wrap items-center gap-2">
      {items.map((item, idx) => (
        <div key={item.id} className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => onSelect(item.id)}
            className={`group relative px-3 py-2 rounded-lg text-sm border ${
              selected === item.id
                ? 'bg-primary text-white border-primary'
                : 'bg-surface-container-low border-outline-variant/25 text-on-surface'
            }`}
          >
            {item.label}
            <span className="pointer-events-none absolute left-1/2 -translate-x-1/2 top-11 w-48 z-20 rounded-md bg-surface-container-highest text-on-surface text-xs p-2 shadow-lg border border-outline-variant/20 opacity-0 group-hover:opacity-100 transition-opacity">
              {item.hint}
            </span>
          </button>
          {idx < items.length - 1 ? <span className="text-on-surface-variant">→</span> : null}
        </div>
      ))}
    </div>
  )
}

function StageCard({
  title,
  selected,
  onSelect,
  what,
  why,
  input,
  process,
  output,
  interpretation,
  next,
}: {
  title: string
  selected: boolean
  onSelect: () => void
  what: string
  why: string
  input: string
  process: string[]
  output: string
  interpretation: string
  next: string
}) {
  return (
    <div className="bg-surface-container-low rounded-xl p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-semibold">{title}</h3>
        <button type="button" onClick={onSelect} className="text-xs underline text-primary">
          {selected ? 'Collapse' : 'Expand'}
        </button>
      </div>
      {selected ? (
        <div className="mt-3 space-y-2 text-sm text-on-surface-variant">
          <p><b>1. What is this?</b> {what}</p>
          <p><b>2. Why is this needed?</b> {why}</p>
          <p><b>3. Input</b> <code>{input}</code></p>
          <div>
            <p><b>4. Process</b></p>
            <ul className="list-disc pl-5">
              {process.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ul>
          </div>
          <p><b>5. Output</b> <code>{output}</code></p>
          <p><b>6. Interpretation</b> {interpretation}</p>
          <p><b>7. What goes to next stage?</b> {next}</p>
        </div>
      ) : null}
    </div>
  )
}

export function Landing() {
  const [mode, setMode] = useState<Mode>('beginner')
  const [selectedStage, setSelectedStage] = useState<StageId>('residual')
  const [showDefinitions, setShowDefinitions] = useState(true)
  const rr = useFetchJson<ReleaseReadiness>(P.releaseReadiness)
  const diff = useFetchJson<DiffDiagnostics>(P.diffDiagnostics)
  const ev = useFetchJson<EvalMetrics>(P.evalMetrics)
  const stageSectionRef = useRef<HTMLDivElement | null>(null)

  const loading = rr.loading || diff.loading || ev.loading
  const error = rr.error ?? diff.error ?? ev.error

  const setMetrics = rr.data?.final_set_level_metrics ?? ev.data?.set_level_metrics
  const tone = useMemo(
    () =>
      mode === 'beginner'
        ? {
            tldrHow: 'Control difficulty (Residual), compare hidden states (SAE), and extract patterns (Diff).',
            tldrConclusion: 'The method captures structure, but needs improvement for prediction.',
          }
        : {
            tldrHow:
              'Compute residual cohorts, encode hidden states with SAE, then run latent diff and set-level evaluation.',
            tldrConclusion:
              'Discovery pipeline shows non-random structure but weak predictive ranking (AUC near random).',
          },
    [mode],
  )

  return (
    <div className="min-h-screen bg-surface text-on-surface font-body">
      <header className="bg-surface fixed top-0 left-0 w-full z-50 flex justify-between items-center px-8 py-3">
        <div className="flex items-center gap-8">
          <span className="text-xl font-bold tracking-tight font-headline">Pipeline Explorer</span>
          <nav className="hidden md:flex gap-6 text-sm">
            <span className="text-primary font-semibold border-b-2 border-primary py-1">Landing</span>
            <Link className="text-on-surface/60 hover:text-on-surface py-1" to="/stage/residual">
              Stage explorer
            </Link>
          </nav>
        </div>
        <div className="flex flex-wrap items-center gap-3 justify-end">
          <ToggleMode mode={mode} onChange={setMode} />
          <Link
            to="/stage/residual"
            className="bg-gradient-to-r from-primary to-primary-container text-white px-6 py-2 rounded-xl font-bold text-sm flex items-center gap-2"
          >
            <MaterialIcon name="hub" />
            Stage explorer
          </Link>
        </div>
      </header>

      <main className="pt-24 px-8 max-w-[1400px] mx-auto pb-20">
        <LoadState loading={loading} error={error}>
          <div className="space-y-8">
            <section className="bg-surface-container-lowest p-8 rounded-2xl shadow-float">
              <p className="text-xs font-bold tracking-widest uppercase text-primary mb-2">TL;DR</p>
              <h1 className="font-headline text-4xl font-extrabold tracking-tight mb-4">
                Residual-Controlled SAE Diffing for LLM failures
              </h1>
              <div className="grid md:grid-cols-2 gap-4 text-sm text-on-surface-variant">
                <p><b>What this page does:</b> Find patterns of failure in LLM predictions.</p>
                <p><b>How:</b> {tone.tldrHow}</p>
                <p>
                  <b>Result:</b> Patterns are found, but predictive power is weak (AUC{' '}
                  {setMetrics?.predictive_utility?.auc?.toFixed(3) ?? '—'}).
                </p>
                <p><b>Conclusion:</b> {tone.tldrConclusion}</p>
              </div>
              <p className="text-xs text-on-surface-variant mt-4">
                How to read this page: start from pipeline, then open stage cards, then read metrics and interpretation.
              </p>
            </section>

            <section className="bg-surface-container-lowest p-6 rounded-2xl">
              <h2 className="font-headline text-xl font-bold mb-3">Pipeline Visualization</h2>
              <PipelineDiagram
                selected={selectedStage}
                onSelect={(stage) => {
                  setSelectedStage(stage)
                  stageSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }}
              />
            </section>

            <section className="bg-surface-container-lowest p-6 rounded-2xl">
              <div className="flex items-center justify-between mb-3">
                <h2 className="font-headline text-xl font-bold">Key Definitions</h2>
                <button
                  type="button"
                  className="text-sm underline text-primary"
                  onClick={() => setShowDefinitions((v) => !v)}
                >
                  {showDefinitions ? 'Hide definitions' : 'Show definitions'}
                </button>
              </div>
              {showDefinitions ? (
                <div className="grid md:grid-cols-2 gap-4 text-sm text-on-surface-variant">
                  <p><ConceptTooltip term="Residual" body="Residual = observed_error - expected_error. High means unusually bad errors." />: observed minus expected error.</p>
                  <p><ConceptTooltip term="Latent" body="A compact feature from SAE encoding that represents hidden-state behavior." />: compressed internal feature representation.</p>
                  <p><ConceptTooltip term="Pattern" body="A recurring activation behavior that differentiates high vs low residual groups." />: repeated model behavior linked to failure.</p>
                  <p><ConceptTooltip term="Diff" body="Statistical comparison between two groups to find separating latent features." />: contrast method for group differences.</p>
                  <p><ConceptTooltip term="AUC" body="0.5 = random ranking, 1.0 = perfect ranking. Current run is near 0.5." />: ranking quality metric.</p>
                  <p><ConceptTooltip term="SAE" body="Sparse Autoencoder turns mixed hidden vectors into more separable features (disentangling)." />: sparse feature extractor over hidden states.</p>
                </div>
              ) : null}
            </section>

            <section ref={stageSectionRef} className="bg-surface-container-lowest p-6 rounded-2xl">
              <h2 className="font-headline text-xl font-bold mb-4">Per-Stage Flow (Structured)</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <StageCard
                  title="Stage: Residual"
                  selected={selectedStage === 'residual'}
                  onSelect={() => setSelectedStage(selectedStage === 'residual' ? 'data' : 'residual')}
                  what="Residual grouping separates unexpected errors from normal difficulty."
                  why="Without this, hard questions and true failures are mixed together."
                  input={'question: "What is photosynthesis?", model_answer: "...", correct: false'}
                  process={[
                    'Estimate expected error per item.',
                    'Compute residual = observed_error - expected_error.',
                    'Group into High Residual and Low Residual.',
                  ]}
                  output="High Residual (n=1964), Low Residual (n=1963)"
                  interpretation="High residual means model is failing worse than expected difficulty."
                  next="Residual groups become labels for latent comparison."
                />
                <StageCard
                  title="Stage: SAE Encode"
                  selected={selectedStage === 'sae'}
                  onSelect={() => setSelectedStage(selectedStage === 'sae' ? 'data' : 'sae')}
                  what="SAE converts dense hidden states into sparse disentangled features."
                  why="Dense vectors are hard to interpret directly for failure patterns."
                  input="Hidden states exported from Gemma rows aligned with residual labels."
                  process={[
                    'Load pooled hidden activations.',
                    'Run SAE encoding.',
                    'Store sparse latent vectors per instance.',
                  ]}
                  output="Latent tensor with d_sae features per row."
                  interpretation="Latents make group differences easier to isolate."
                  next="Latents feed diff mining."
                />
                <StageCard
                  title="Stage: Diff"
                  selected={selectedStage === 'diff'}
                  onSelect={() => setSelectedStage(selectedStage === 'diff' ? 'data' : 'diff')}
                  what="Diff finds latent features that separate High vs Low residual."
                  why="We need concrete candidate patterns, not just aggregate metrics."
                  input="SAE latents + residual group labels."
                  process={[
                    'Run group-wise statistical tests.',
                    'Rank candidates by separation quality.',
                    'Build pattern catalog and membership.',
                  ]}
                  output={`${diff.data?.n_latents_tested?.toLocaleString() ?? '—'} tested, ${diff.data?.n_passing?.toLocaleString() ?? '—'} passing patterns`}
                  interpretation="Passing patterns indicate non-random structure."
                  next="Pattern memberships move to evaluation."
                />
                <StageCard
                  title="Stage: Eval"
                  selected={selectedStage === 'eval'}
                  onSelect={() => setSelectedStage(selectedStage === 'eval' ? 'data' : 'eval')}
                  what="Eval checks if discovered patterns are predictive and non-redundant."
                  why="Many patterns can exist but still fail to predict errors."
                  input="Pattern catalog + per-instance memberships."
                  process={[
                    'Compute AUC/F1 for predictive utility.',
                    'Measure overlap using Jaccard.',
                    'Summarize coverage and concentration.',
                  ]}
                  output={`AUC ${setMetrics?.predictive_utility?.auc?.toFixed(3) ?? '—'}, F1 ${setMetrics?.predictive_utility?.f1?.toFixed(3) ?? '—'}`}
                  interpretation="Current patterns capture structure but weakly rank failures."
                  next="Find causes of weak AUC and improve the method."
                />
              </div>
            </section>

            <section className="bg-surface-container-lowest p-6 rounded-2xl">
              <h2 className="font-headline text-xl font-bold mb-3">Result Section</h2>
              <MetricTable
                auc={setMetrics?.predictive_utility?.auc}
                f1={setMetrics?.predictive_utility?.f1}
                jaccard={setMetrics?.redundancy?.avg_pairwise_jaccard}
              />
              <div className="bg-surface-container-low rounded-xl p-4 mt-4 text-sm">
                <p className="font-semibold mb-2">Interpretation</p>
                <p>✔ We found patterns (non-random structure).</p>
                <p>✘ But patterns are not useful enough for prediction.</p>
                <p className="mt-2"><b>Conclusion:</b> Method works partially, but not enough.</p>
              </div>
            </section>

            <section className="grid md:grid-cols-2 gap-4">
              <ExampleBox title="Before/After Example">
                {'Without method:\n→ random grouping\n\nWith method:\n→ structured but weak predictive power'}
              </ExampleBox>
              <ExampleBox title="Pattern Example">
                {'Pattern #23:\n→ Activated when question involves biology terms\n→ Appears mostly in wrong answers\n\nInterpretation:\n→ Model struggles with biology reasoning'}
              </ExampleBox>
              <ExampleBox title="Data Flow Traceability">
                {'Instance #123:\n→ residual = 0.8\n→ latent pattern activated: [12, 45]\n→ classified as failure'}
              </ExampleBox>
              <ExampleBox title="Why AUC is low">
                {'- Patterns overlap too much\n- Not discriminative enough\n- Noise in latent space'}
              </ExampleBox>
            </section>

            <section className="bg-surface-container-lowest p-6 rounded-2xl">
              <h2 className="font-headline text-xl font-bold mb-3">Next Steps (Actionable)</h2>
              <div className="space-y-2 text-sm">
                <label className="flex items-center gap-2"><input type="checkbox" disabled /> Run multi-seed stability</label>
                <label className="flex items-center gap-2"><input type="checkbox" disabled /> Reduce pattern overlap</label>
                <label className="flex items-center gap-2"><input type="checkbox" disabled /> Tune SAE width</label>
                <label className="flex items-center gap-2"><input type="checkbox" disabled /> Add baseline comparison</label>
              </div>
            </section>

            <section className="bg-surface-container-low p-6 rounded-2xl">
              <h2 className="font-headline text-xl font-bold mb-3">Visual References</h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <img src={P.residualPng} alt="Residual by subject" className="w-full rounded-lg bg-surface" />
                  <p className="text-xs text-on-surface-variant mt-2">Residual-by-subject concentration map.</p>
                </div>
                <div>
                  <img src={P.evalResidualBar} alt="Residual evaluation summary" className="w-full rounded-lg bg-surface" />
                  <p className="text-xs text-on-surface-variant mt-2">Final evaluation comparison across residual groups.</p>
                </div>
              </div>
            </section>
          </div>
        </LoadState>
      </main>
    </div>
  )
}
