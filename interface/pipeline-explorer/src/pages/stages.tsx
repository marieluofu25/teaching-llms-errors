import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
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

type Lang = 'en' | 'id'
type StageId = 'residual' | 'export' | 'sae' | 'diff' | 'eval' | 'report'

type LangText = { en: string; id: string }
type KeyConcept = { term: LangText; definition: LangText; formula?: LangText; example?: LangText }
type ArtifactSpec = { name: LangText; file: string; contains: LangText; usedBy: LangText }
type StageCopy = {
  step: LangText
  title: LangText
  description: LangText
  overview: LangText[]
  keyConcepts: KeyConcept[]
  methodSteps: LangText[]
  interpretationRules: LangText[]
  insights: LangText[]
  concreteExample: LangText[]
  resultsGuide: LangText[]
  artifacts: ArtifactSpec[]
}

const STAGE_ORDER: { id: StageId; path: string }[] = [
  { id: 'residual', path: '/stage/residual' },
  { id: 'export', path: '/stage/export' },
  { id: 'sae', path: '/stage/sae' },
  { id: 'diff', path: '/stage/diff' },
  { id: 'eval', path: '/stage/eval' },
  { id: 'report', path: '/stage/report' },
]

function t(text: LangText, lang: Lang) {
  return text[lang]
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-surface-container-lowest rounded-xl p-6">
      <h3 className="font-headline font-bold mb-3">{title}</h3>
      {children}
    </section>
  )
}

function StageLanguageToggle({ lang, setLang }: { lang: Lang; setLang: (v: Lang) => void }) {
  return (
    <div className="rounded-xl border border-outline-variant/30 p-1 inline-flex gap-1">
      <button
        type="button"
        onClick={() => setLang('en')}
        className={`px-3 py-1 text-xs rounded-lg font-semibold ${lang === 'en' ? 'bg-primary text-white' : 'text-on-surface-variant'}`}
      >
        EN
      </button>
      <button
        type="button"
        onClick={() => setLang('id')}
        className={`px-3 py-1 text-xs rounded-lg font-semibold ${lang === 'id' ? 'bg-primary text-white' : 'text-on-surface-variant'}`}
      >
        ID
      </button>
    </div>
  )
}

function StageArtifacts({ artifacts, lang }: { artifacts: ArtifactSpec[]; lang: Lang }) {
  return (
    <div className="rounded-xl bg-surface-container-low p-4">
      <p className="text-xs font-bold uppercase tracking-widest text-on-surface-variant/70">
        {lang === 'en' ? 'Gemma pipeline artifacts' : 'Artifact pipeline Gemma'}
      </p>
      <div className="mt-3 space-y-3">
        {artifacts.map((a) => (
          <div key={a.file} className="bg-surface-container-lowest rounded-lg p-3 text-sm">
            <p className="font-semibold">{t(a.name, lang)}</p>
            <p className="text-on-surface-variant">{t(a.contains, lang)}</p>
            <p className="text-xs mt-1 text-on-surface-variant">
              {lang === 'en' ? 'Used by:' : 'Dipakai oleh:'} {t(a.usedBy, lang)}
            </p>
            <code className="text-xs bg-surface-container-low px-1.5 py-0.5 rounded inline-block mt-1">{a.file}</code>
          </div>
        ))}
      </div>
    </div>
  )
}

function StageHeader({
  step,
  accent,
  title,
  description,
  rightSlot,
}: {
  step: string
  accent: string
  title: string
  description: string
  rightSlot?: React.ReactNode
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
      {rightSlot}
    </div>
  )
}

function StageNavBlock({ activeId, lang }: { activeId: StageId; lang: Lang }) {
  const idx = STAGE_ORDER.findIndex((s) => s.id === activeId)
  const prev = idx > 0 ? STAGE_ORDER[idx - 1] : null
  const next = idx < STAGE_ORDER.length - 1 ? STAGE_ORDER[idx + 1] : null
  return (
    <div className="flex justify-between gap-3 mt-6">
      {prev ? (
        <Link to={prev.path} className="px-4 py-2 rounded-lg bg-surface-container-low hover:bg-surface-container-high">
          {lang === 'en' ? 'Previous stage' : 'Stage sebelumnya'}
        </Link>
      ) : <span />}
      {next ? (
        <Link to={next.path} className="px-4 py-2 rounded-lg bg-primary text-white hover:bg-primary-container">
          {lang === 'en' ? 'Next stage' : 'Stage berikutnya'}
        </Link>
      ) : null}
    </div>
  )
}

function GuidelineSections({ copy, lang }: { copy: StageCopy; lang: Lang }) {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
      <Section title={lang === 'en' ? 'Overview' : 'Overview'}>
        <ul className="list-disc pl-4 text-sm space-y-1">
          {copy.overview.map((x, i) => <li key={i}>{t(x, lang)}</li>)}
        </ul>
      </Section>
      <Section title={lang === 'en' ? 'Key Concepts' : 'Konsep Kunci'}>
        <div className="space-y-3 text-sm">
          {copy.keyConcepts.map((k, i) => (
            <div key={i} className="bg-surface-container-low rounded-lg p-3">
              <p><strong>{t(k.term, lang)}:</strong> {t(k.definition, lang)}</p>
              {k.formula ? <p className="text-xs mt-1"><strong>{lang === 'en' ? 'Formula:' : 'Rumus:'}</strong> {t(k.formula, lang)}</p> : null}
              {k.example ? <p className="text-xs mt-1"><strong>{lang === 'en' ? 'Example:' : 'Contoh:'}</strong> {t(k.example, lang)}</p> : null}
            </div>
          ))}
        </div>
      </Section>
      <Section title={lang === 'en' ? 'Method / Process' : 'Metode / Proses'}>
        <ol className="list-decimal pl-4 text-sm space-y-1">
          {copy.methodSteps.map((x, i) => <li key={i}>{t(x, lang)}</li>)}
        </ol>
      </Section>
      <Section title={lang === 'en' ? 'How to Read Results' : 'Cara Membaca Hasil'}>
        <ul className="list-disc pl-4 text-sm space-y-1">
          {copy.interpretationRules.map((x, i) => <li key={i}>{t(x, lang)}</li>)}
        </ul>
      </Section>
      <Section title={lang === 'en' ? 'Results Guide' : 'Panduan Hasil'}>
        <ul className="list-disc pl-4 text-sm space-y-1">
          {copy.resultsGuide.map((x, i) => <li key={i}>{t(x, lang)}</li>)}
        </ul>
      </Section>
      <Section title={lang === 'en' ? 'Key Insights' : 'Insight Kunci'}>
        <ul className="list-disc pl-4 text-sm space-y-1">
          {copy.insights.map((x, i) => <li key={i}>{t(x, lang)}</li>)}
        </ul>
      </Section>
      <Section title={lang === 'en' ? 'Concrete Example' : 'Contoh Konkret'}>
        <ul className="list-disc pl-4 text-sm space-y-1">
          {copy.concreteExample.map((x, i) => <li key={i}>{t(x, lang)}</li>)}
        </ul>
      </Section>
      <Section title={lang === 'en' ? 'Output Artifacts' : 'Artifact Output'}>
        <StageArtifacts artifacts={copy.artifacts} lang={lang} />
      </Section>
    </div>
  )
}

function stageCopyFor(stageId: StageId): StageCopy {
  const commonArtifacts = {
    report: { en: 'report stage', id: 'stage report' },
    eval: { en: 'eval stage', id: 'stage eval' },
    next: { en: 'next stage', id: 'stage berikutnya' },
  }
  const all: Record<StageId, StageCopy> = {
    residual: {
      step: { en: 'Stage 01: Residuals', id: 'Stage 01: Residual' },
      title: { en: 'Residual analysis — error by subject', id: 'Analisis residual — error per subjek' },
      description: {
        en: 'Process: estimate expected error, compute residual, then group errors into high/low/neutral cohorts.',
        id: 'Proses: estimasi expected error, hitung residual, lalu kelompokkan error ke cohort high/low/neutral.',
      },
      overview: [
        { en: 'This stage checks where Gemma fails more than expected difficulty.', id: 'Stage ini mengecek area Gemma gagal lebih buruk dari difficulty yang diharapkan.' },
        { en: 'Residual helps separate true failure patterns from generic hard questions.', id: 'Residual membantu memisahkan pola gagal nyata dari soal yang memang sulit.' },
        { en: 'Main outputs are residual profile, group counts, and subject-level error plot.', id: 'Output utama adalah residual profile, jumlah group, dan plot error per subjek.' },
      ],
      keyConcepts: [
        { term: { en: 'Observed Error', id: 'Observed Error' }, definition: { en: 'Actual model outcome: 1 if wrong, 0 if correct.', id: 'Hasil aktual model: 1 jika salah, 0 jika benar.' }, example: { en: 'Model answers wrong -> observed error = 1.', id: 'Model menjawab salah -> observed error = 1.' } },
        { term: { en: 'Expected Error', id: 'Expected Error' }, definition: { en: 'Predicted probability of error from lightweight features.', id: 'Probabilitas error yang diprediksi dari fitur ringan.' }, example: { en: 'Estimated expected error = 0.70.', id: 'Expected error terestimasi = 0.70.' } },
        { term: { en: 'Residual', id: 'Residual' }, definition: { en: 'Gap between observed and expected error.', id: 'Selisih antara observed dan expected error.' }, formula: { en: 'residual = observed - expected', id: 'residual = observed - expected' }, example: { en: '1 - 0.7 = 0.3, indicating worse-than-expected behavior.', id: '1 - 0.7 = 0.3, menandakan performa lebih buruk dari ekspektasi.' } },
      ],
      methodSteps: [
        { en: 'Estimate expected error for each MMLU row using lightweight predictors.', id: 'Estimasi expected error untuk tiap baris MMLU dengan prediktor ringan.' },
        { en: 'Compute residual for each row.', id: 'Hitung residual untuk tiap baris.' },
        { en: 'Assign residual groups (high, low, neutral) by threshold rule.', id: 'Tetapkan group residual (high, low, neutral) berdasarkan threshold.' },
        { en: 'Aggregate by subject and compare error concentration.', id: 'Agregasi per subjek dan bandingkan konsentrasi error.' },
      ],
      interpretationRules: [
        { en: 'If high residual is large, errors are not explained by difficulty alone.', id: 'Jika high residual besar, error tidak cukup dijelaskan oleh difficulty.' },
        { en: 'If low residual dominates, model may handle hard questions better than expected.', id: 'Jika low residual dominan, model mungkin menangani soal sulit lebih baik dari ekspektasi.' },
        { en: 'If a subject has high error rate and many high-residual rows, prioritize it for analysis.', id: 'Jika subjek punya error rate tinggi dan banyak high-residual, prioritaskan untuk analisis.' },
      ],
      resultsGuide: [
        { en: 'Error-by-subject chart shows where mistakes are concentrated.', id: 'Chart error-per-subjek menunjukkan konsentrasi kesalahan.' },
        { en: 'Residual-group bars show distribution of high/low/neutral cohorts.', id: 'Bar residual-group menunjukkan distribusi cohort high/low/neutral.' },
        { en: 'Top-subject table helps rank investigation priority.', id: 'Tabel top-subjek membantu ranking prioritas investigasi.' },
      ],
      insights: [
        { en: 'Residual grouping is the foundation for all downstream contrast analysis.', id: 'Residual grouping adalah fondasi semua analisis kontras berikutnya.' },
        { en: 'Subject-level concentration quickly highlights risky domains.', id: 'Konsentrasi per subjek cepat menyorot domain berisiko.' },
        { en: 'This stage converts raw error logs into interpretable cohorts.', id: 'Stage ini mengubah log error mentah menjadi cohort yang bisa diinterpretasi.' },
      ],
      concreteExample: [
        { en: 'Input: expected error = 0.70, model answer is wrong (observed = 1).', id: 'Input: expected error = 0.70, jawaban model salah (observed = 1).' },
        { en: 'Compute: residual = 1 - 0.70 = 0.30.', id: 'Hitung: residual = 1 - 0.70 = 0.30.' },
        { en: 'Output: row enters High Residual cohort.', id: 'Output: baris masuk cohort High Residual.' },
      ],
      artifacts: [
        { name: { en: 'Merged Gemma predictions', id: 'Prediksi Gemma gabungan' }, file: 'stage2/s01_residual/results/mmlu_gemma_predictions_merged.csv', contains: { en: 'Row-aligned predictions used to compute residuals.', id: 'Prediksi per baris untuk menghitung residual.' }, usedBy: commonArtifacts.next },
        { name: { en: 'Residual profile', id: 'Residual profile' }, file: 'stage2/s01_residual/results/mmlu_full_residuals_profile.json', contains: { en: 'Global and subject-level residual statistics.', id: 'Statistik residual global dan per subjek.' }, usedBy: commonArtifacts.next },
        { name: { en: 'Residual metadata', id: 'Metadata residual' }, file: 'stage2/s01_residual/results/mmlu_full_residuals.meta.json', contains: { en: 'Group counts and summary metadata.', id: 'Jumlah group dan metadata ringkas.' }, usedBy: commonArtifacts.next },
      ],
    },
    export: {
      step: { en: 'Stage 02: Export', id: 'Stage 02: Export' }, title: { en: 'HF hidden activations', id: 'Aktivasi hidden HF' },
      description: { en: 'Process: export pooled hidden states from Gemma for each residual-aligned row.', id: 'Proses: ekspor hidden state Gemma yang sudah selaras residual per baris.' },
      overview: [{ en: 'This stage bridges behavior logs to model internal states.', id: 'Stage ini menjembatani log perilaku ke state internal model.' }, { en: 'It keeps exact row alignment with residual outputs.', id: 'Stage ini menjaga alignment baris yang sama dengan output residual.' }, { en: 'Main outputs are hidden metadata and hidden NPZ pointer.', id: 'Output utama berupa metadata hidden dan pointer NPZ hidden.' }],
      keyConcepts: [{ term: { en: 'Hidden state', id: 'Hidden state' }, definition: { en: 'Vector representation from a model layer for one input row.', id: 'Vektor representasi dari layer model untuk satu baris input.' } }, { term: { en: 'Layer index', id: 'Layer index' }, definition: { en: 'Chosen transformer layer used for export.', id: 'Layer transformer yang dipilih saat export.' } }, { term: { en: 'Row alignment', id: 'Row alignment' }, definition: { en: 'Exported vectors keep the same order as residual rows.', id: 'Urutan vektor hasil export sama dengan urutan baris residual.' } }],
      methodSteps: [{ en: 'Load residual-aligned rows and selected model layer.', id: 'Muat baris yang align residual dan layer model terpilih.' }, { en: 'Extract pooled hidden vector per row.', id: 'Ekstrak pooled hidden vector per baris.' }, { en: 'Save NPZ tensor and metadata file.', id: 'Simpan tensor NPZ dan file metadata.' }],
      interpretationRules: [{ en: 'If rows match residual count, downstream join is safe.', id: 'Jika jumlah baris cocok dengan residual, join downstream aman.' }, { en: 'If hidden dim is unexpected, check model/layer configuration.', id: 'Jika hidden dim tidak sesuai, cek konfigurasi model/layer.' }, { en: 'If output NPZ path exists, SAE stage can start.', id: 'Jika path output NPZ ada, stage SAE bisa jalan.' }],
      resultsGuide: [{ en: 'Model and tensor panel summarizes export health.', id: 'Panel model/tensor merangkum kesehatan export.' }, { en: 'Output NPZ name tells which file SAE consumes next.', id: 'Nama output NPZ menunjukkan file yang dikonsumsi SAE berikutnya.' }],
      insights: [{ en: 'Export quality is mostly about shape and row consistency.', id: 'Kualitas export terutama soal shape dan konsistensi baris.' }, { en: 'This stage does not judge quality of patterns yet.', id: 'Stage ini belum menilai kualitas pattern.' }, { en: 'A clean export prevents cascading errors in SAE/diff.', id: 'Export yang bersih mencegah error berantai di SAE/diff.' }],
      concreteExample: [{ en: 'Input: one residual row at index 128.', id: 'Input: satu baris residual di index 128.' }, { en: 'Process: read layer activation and pool to 1 vector.', id: 'Proses: baca aktivasi layer lalu pooling jadi 1 vektor.' }, { en: 'Output: row 128 in hidden NPZ.', id: 'Output: baris 128 di hidden NPZ.' }],
      artifacts: [
        { name: { en: 'Hidden export metadata', id: 'Metadata hidden export' }, file: 'stage2/s02_export/results/mmlu_full_hidden.meta.json', contains: { en: 'Model name, row count, hidden dim, layer index.', id: 'Nama model, jumlah baris, hidden dim, layer index.' }, usedBy: { en: 'SAE stage', id: 'stage SAE' } },
        { name: { en: 'Hidden tensor', id: 'Tensor hidden' }, file: 'stage2/s02_export/results/mmlu_full_hidden.npz', contains: { en: 'Dense hidden vectors per row.', id: 'Vektor hidden padat per baris.' }, usedBy: { en: 'SAE stage', id: 'stage SAE' } },
      ],
    },
    sae: {
      step: { en: 'Stage 03: SAE encode', id: 'Stage 03: SAE encode' }, title: { en: 'Sparse autoencoder latents', id: 'Latent sparse autoencoder' },
      description: { en: 'Process: encode dense hidden vectors into sparse latent features.', id: 'Proses: encode vektor hidden padat ke fitur latent yang sparse.' },
      overview: [{ en: 'This stage converts dense representations into interpretable sparse features.', id: 'Stage ini mengubah representasi padat menjadi fitur sparse yang lebih interpretabel.' }, { en: 'Sparse latents make group contrast easier in diff stage.', id: 'Latent sparse memudahkan kontras group di stage diff.' }, { en: 'Main outputs are SAE metadata and latent NPZ pointer.', id: 'Output utama berupa metadata SAE dan pointer latent NPZ.' }],
      keyConcepts: [{ term: { en: 'SAE', id: 'SAE' }, definition: { en: 'Sparse autoencoder that maps dense activations to sparse features.', id: 'Sparse autoencoder yang memetakan aktivasi padat ke fitur sparse.' } }, { term: { en: 'Latent feature', id: 'Fitur latent' }, definition: { en: 'One interpretable activation dimension in SAE space.', id: 'Satu dimensi aktivasi yang lebih interpretabel di ruang SAE.' } }, { term: { en: 'd_sae', id: 'd_sae' }, definition: { en: 'Total latent dimensions produced by the SAE.', id: 'Total dimensi latent yang diproduksi SAE.' } }],
      methodSteps: [{ en: 'Load hidden NPZ from export stage.', id: 'Muat hidden NPZ dari stage export.' }, { en: 'Apply SAELens encoder to each row vector.', id: 'Terapkan encoder SAELens ke setiap vektor baris.' }, { en: 'Save latent NPZ and metadata for diff stage.', id: 'Simpan latent NPZ dan metadata untuk stage diff.' }],
      interpretationRules: [{ en: 'If d_sae is high, feature space is more granular.', id: 'Jika d_sae besar, ruang fitur lebih granular.' }, { en: 'If row counts match export, latent mapping is aligned.', id: 'Jika jumlah baris cocok dengan export, mapping latent align.' }, { en: 'If SAE id/release mismatch expected config, results may shift.', id: 'Jika SAE id/release tidak sesuai konfigurasi, hasil bisa bergeser.' }],
      resultsGuide: [{ en: 'Metadata rows confirm shape and model compatibility.', id: 'Baris metadata memastikan shape dan kompatibilitas model.' }, { en: 'Latent NPZ is the direct input for diff mining.', id: 'Latent NPZ adalah input langsung untuk diff mining.' }],
      insights: [{ en: 'SAE stage prepares the representation for statistical diffing.', id: 'Stage SAE menyiapkan representasi untuk diff statistik.' }, { en: 'Most checks here are structural, not predictive.', id: 'Sebagian besar cek di sini bersifat struktural, bukan prediktif.' }, { en: 'Consistent row alignment is non-negotiable.', id: 'Konsistensi alignment baris wajib dijaga.' }],
      concreteExample: [{ en: 'Input: dense hidden vector h for one row.', id: 'Input: vektor hidden padat h untuk satu baris.' }, { en: 'Process: SAE maps h to sparse z.', id: 'Proses: SAE memetakan h menjadi z yang sparse.' }, { en: 'Output: row latent activations saved in NPZ.', id: 'Output: aktivasi latent per baris disimpan di NPZ.' }],
      artifacts: [
        { name: { en: 'SAE latent metadata', id: 'Metadata latent SAE' }, file: 'stage2/s03_sae_encode/results/mmlu_full_sae_latents.meta.json', contains: { en: 'SAE release/id, latent dimension, row count.', id: 'SAE release/id, dimensi latent, jumlah baris.' }, usedBy: { en: 'diff stage', id: 'stage diff' } },
        { name: { en: 'SAE latent tensor', id: 'Tensor latent SAE' }, file: 'stage2/s03_sae_encode/results/mmlu_full_sae_latents.npz', contains: { en: 'Sparse latent activations per row.', id: 'Aktivasi latent sparse per baris.' }, usedBy: { en: 'diff stage', id: 'stage diff' } },
      ],
    },
    diff: {
      step: { en: 'Stage 04: Diff', id: 'Stage 04: Diff' }, title: { en: 'Latent diff & patterns', id: 'Diff latent & pattern' },
      description: { en: 'Process: compare high vs low residual cohorts to rank separating latent patterns.', id: 'Proses: membandingkan cohort high vs low residual untuk ranking pattern latent pemisah.' },
      overview: [{ en: 'This stage searches for candidate failure signatures in latent space.', id: 'Stage ini mencari kandidat signature kegagalan di latent space.' }, { en: 'It is a statistical contrast, not direct causal proof.', id: 'Ini adalah kontras statistik, bukan bukti kausal langsung.' }, { en: 'Main outputs are diagnostics and pattern catalog.', id: 'Output utama adalah diagnostics dan pattern catalog.' }],
      keyConcepts: [{ term: { en: "V'", id: "V'" }, definition: { en: 'Diff effect score for latent separation strength.', id: 'Skor efek diff untuk kekuatan pemisahan latent.' } }, { term: { en: 'p-value', id: 'p-value' }, definition: { en: 'Statistical significance estimate for observed difference.', id: 'Estimasi signifikansi statistik untuk perbedaan yang diamati.' } }, { term: { en: 'threshold', id: 'threshold' }, definition: { en: 'Decision boundary used for pattern membership.', id: 'Batas keputusan untuk membership pattern.' } }],
      methodSteps: [{ en: 'Split rows into high and low residual groups.', id: 'Pisahkan baris ke group high dan low residual.' }, { en: 'Test each latent for group separation signal.', id: 'Uji tiap latent untuk sinyal pemisahan group.' }, { en: 'Filter/rank candidates and export pattern catalog.', id: 'Filter/ranking kandidat lalu ekspor pattern catalog.' }],
      interpretationRules: [{ en: "If V' is high and p-value is small, the latent is a stronger candidate.", id: "Jika V' tinggi dan p-value kecil, latent kandidat lebih kuat." }, { en: 'If passing count is low, criteria may be too strict or signal weak.', id: 'Jika passing count rendah, kriteria bisa terlalu ketat atau sinyal lemah.' }, { en: 'If many similar patterns appear, redundancy should be checked in eval.', id: 'Jika banyak pattern mirip, redundansi harus dicek di eval.' }],
      resultsGuide: [{ en: 'KPI cards show cohort sizes and pass volume.', id: 'Kartu KPI menunjukkan ukuran cohort dan jumlah pattern lolos.' }, { en: 'Pattern table ranks candidate features for follow-up evaluation.', id: 'Tabel pattern meranking fitur kandidat untuk evaluasi lanjutan.' }],
      insights: [{ en: 'Diff stage generates hypotheses for failure structure.', id: 'Stage diff menghasilkan hipotesis struktur kegagalan.' }, { en: 'Top-ranked patterns are candidates, not final truths.', id: 'Pattern top-ranked adalah kandidat, bukan kebenaran final.' }, { en: 'Eval stage decides whether candidates are useful globally.', id: 'Stage eval menentukan apakah kandidat berguna secara global.' }],
      concreteExample: [{ en: 'Input: latent 452 values in high vs low groups.', id: 'Input: nilai latent 452 pada group high vs low.' }, { en: 'Result: V\'=0.31, p=1.2e-4, threshold=0.11.', id: 'Hasil: V\'=0.31, p=1.2e-4, threshold=0.11.' }, { en: 'Output: latent becomes a passing candidate pattern.', id: 'Output: latent menjadi pattern kandidat yang lolos.' }],
      artifacts: [
        { name: { en: 'Diff diagnostics', id: 'Diagnostics diff' }, file: 'stage2/s04_diff/results/mmlu_full_latents.diagnostics.json', contains: { en: 'Counts for tested/passing latents and cohort sizes.', id: 'Jumlah latent diuji/lolos dan ukuran cohort.' }, usedBy: { en: 'eval stage', id: 'stage eval' } },
        { name: { en: 'Pattern catalog', id: 'Pattern catalog' }, file: 'stage2/s04_diff/results/mmlu_full_latents.pattern_catalog.json', contains: { en: 'Ranked candidate patterns with scores.', id: 'Pattern kandidat berperingkat beserta skor.' }, usedBy: { en: 'eval stage', id: 'stage eval' } },
        { name: { en: 'Pattern membership', id: 'Membership pattern' }, file: 'stage2/s04_diff/results/mmlu_full_latents.pattern_membership.csv', contains: { en: 'Per-row pattern membership signals.', id: 'Sinyal membership pattern per baris.' }, usedBy: { en: 'eval stage', id: 'stage eval' } },
      ],
    },
    eval: {
      step: { en: 'Stage 05: Eval', id: 'Stage 05: Eval' }, title: { en: 'Metrics & leaderboard', id: 'Metrik & leaderboard' },
      description: { en: 'Process: evaluate pattern set quality with set-level metrics and leaderboard summary.', id: 'Proses: evaluasi kualitas pattern set dengan metrik set-level dan ringkasan leaderboard.' },
      overview: [{ en: 'This stage checks whether discovered patterns are actually useful.', id: 'Stage ini mengecek apakah pattern yang ditemukan benar-benar berguna.' }, { en: 'It measures coverage, utility, and redundancy at set level.', id: 'Stage ini mengukur coverage, utility, dan redundansi pada level set.' }, { en: 'Main outputs are final metrics and leaderboard snapshot.', id: 'Output utama adalah final metrics dan snapshot leaderboard.' }],
      keyConcepts: [{ term: { en: 'Coverage', id: 'Coverage' }, definition: { en: 'How many errors are touched by the pattern set.', id: 'Seberapa banyak error yang terjangkau oleh pattern set.' } }, { term: { en: 'AUC/F1', id: 'AUC/F1' }, definition: { en: 'Predictive utility metrics for ranking/classification quality.', id: 'Metrik utility prediktif untuk kualitas ranking/klasifikasi.' } }, { term: { en: 'Redundancy', id: 'Redundansi' }, definition: { en: 'Overlap among patterns, often measured by Jaccard.', id: 'Overlap antarpattern, sering diukur dengan Jaccard.' } }],
      methodSteps: [{ en: 'Load pattern membership and residual labels.', id: 'Muat membership pattern dan label residual.' }, { en: 'Compute set-level metrics (coverage, utility, redundancy, stability).', id: 'Hitung metrik set-level (coverage, utility, redundansi, stabilitas).' }, { en: 'Export metrics JSON, leaderboard, and residual summary chart.', id: 'Ekspor metrics JSON, leaderboard, dan chart ringkasan residual.' }],
      interpretationRules: [{ en: 'If AUC is near 0.5, ranking signal is weak.', id: 'Jika AUC dekat 0.5, sinyal ranking lemah.' }, { en: 'If coverage is high but concentration is low, patterns may be broad but noisy.', id: 'Jika coverage tinggi tapi concentration rendah, pattern mungkin luas tapi noisy.' }, { en: 'If Jaccard is high, pattern set has redundancy risk.', id: 'Jika Jaccard tinggi, pattern set berisiko redundan.' }],
      resultsGuide: [{ en: 'Metric cards summarize the quick health check.', id: 'Kartu metrik merangkum health check cepat.' }, { en: 'Leaderboard table compares run-level summary rows.', id: 'Tabel leaderboard membandingkan ringkasan per run.' }, { en: 'Residual bar plot links cohort behavior to final metrics.', id: 'Bar plot residual menghubungkan perilaku cohort ke metrik final.' }],
      insights: [{ en: 'Eval turns candidate patterns into measurable evidence.', id: 'Eval mengubah pattern kandidat menjadi bukti terukur.' }, { en: 'Utility and redundancy should be interpreted together.', id: 'Utility dan redundansi harus dibaca bersama.' }, { en: 'Low utility indicates clear next-step optimization targets.', id: 'Utility rendah menunjukkan target optimasi langkah berikutnya.' }],
      concreteExample: [{ en: 'Input: pattern memberships from diff stage.', id: 'Input: membership pattern dari stage diff.' }, { en: 'Observed: AUC=0.50, F1=0.57, Jaccard=0.60.', id: 'Teramati: AUC=0.50, F1=0.57, Jaccard=0.60.' }, { en: 'Interpretation: class signal exists but ranking quality is still weak.', id: 'Interpretasi: sinyal kelas ada tetapi kualitas ranking masih lemah.' }],
      artifacts: [
        { name: { en: 'Final metrics', id: 'Metrik final' }, file: 'stage2/s05_eval/results/mmlu_final_metrics.json', contains: { en: 'Set-level metrics and residual metrics.', id: 'Metrik set-level dan metrik residual.' }, usedBy: commonArtifacts.report },
        { name: { en: 'Leaderboard snapshot', id: 'Snapshot leaderboard' }, file: 'stage2/s05_eval/results/leaderboard.csv', contains: { en: 'Compact comparison row(s) for this run.', id: 'Baris perbandingan ringkas untuk run ini.' }, usedBy: commonArtifacts.report },
        { name: { en: 'Residual summary plot', id: 'Plot ringkasan residual' }, file: 'stage2/s05_eval/results/mmlu_gemma_unified_sae_residual_bar.png', contains: { en: 'Visual of residual-stratified performance.', id: 'Visual performa berdasarkan stratifikasi residual.' }, usedBy: commonArtifacts.report },
      ],
    },
    report: {
      step: { en: 'Stage 06: Report', id: 'Stage 06: Report' }, title: { en: 'Release readiness & HTML report', id: 'Release readiness & laporan HTML' },
      description: { en: 'Process: aggregate gates and compile final report for communication and handoff.', id: 'Proses: agregasi gate dan kompilasi laporan final untuk komunikasi dan handoff.' },
      overview: [{ en: 'This final stage packages outputs into a decision-friendly report.', id: 'Stage final ini mengemas output menjadi laporan yang mudah dipakai untuk keputusan.' }, { en: 'It includes readiness verdict and violation context.', id: 'Isinya mencakup verdict readiness dan konteks pelanggaran.' }, { en: 'Main outputs are release_readiness.json and mmlu_report.html.', id: 'Output utama adalah release_readiness.json dan mmlu_report.html.' }],
      keyConcepts: [{ term: { en: 'Release readiness', id: 'Release readiness' }, definition: { en: 'Gate-based decision summary for run quality.', id: 'Ringkasan keputusan berbasis gate untuk kualitas run.' } }, { term: { en: 'Gate verdict', id: 'Gate verdict' }, definition: { en: 'Pass/fail status from quality thresholds.', id: 'Status lulus/gagal dari threshold kualitas.' } }, { term: { en: 'Violation detail', id: 'Detail pelanggaran' }, definition: { en: 'Specific reason a gate failed.', id: 'Alasan spesifik kenapa gate gagal.' } }],
      methodSteps: [{ en: 'Read all upstream stage outputs.', id: 'Baca semua output stage sebelumnya.' }, { en: 'Apply gate thresholds and compute verdict.', id: 'Terapkan threshold gate dan hitung verdict.' }, { en: 'Render JSON summary and HTML narrative report.', id: 'Render ringkasan JSON dan laporan HTML naratif.' }],
      interpretationRules: [{ en: 'If verdict is pass, outputs meet defined quality gates.', id: 'Jika verdict pass, output memenuhi gate kualitas yang ditetapkan.' }, { en: 'If verdict is fail, check violations before downstream use.', id: 'Jika verdict fail, cek violations sebelum dipakai lanjut.' }, { en: 'HTML report is for communication; JSON is for programmatic checks.', id: 'Laporan HTML untuk komunikasi; JSON untuk cek terprogram.' }],
      resultsGuide: [{ en: 'Embedded preview shows the final report readers will see.', id: 'Preview embedded menampilkan laporan final yang akan dibaca user.' }, { en: 'JSON highlights expose machine-readable gate details.', id: 'JSON highlights menampilkan detail gate yang machine-readable.' }],
      insights: [{ en: 'Report stage is the final trust layer for stakeholders.', id: 'Stage report adalah lapisan kepercayaan akhir untuk stakeholder.' }, { en: 'A failed gate is a quality signal, not a research dead end.', id: 'Gate gagal adalah sinyal kualitas, bukan akhir riset.' }, { en: 'This stage supports reproducible handoff and auditing.', id: 'Stage ini mendukung handoff dan audit yang reproducible.' }],
      concreteExample: [{ en: 'Input: eval metrics with low AUC and a gate threshold.', id: 'Input: metrik eval dengan AUC rendah dan threshold gate.' }, { en: 'Process: threshold check marks violation.', id: 'Proses: pengecekan threshold menandai violation.' }, { en: 'Output: verdict=no-go with violation detail in JSON and HTML.', id: 'Output: verdict=no-go dengan detail violation di JSON dan HTML.' }],
      artifacts: [
        { name: { en: 'Release readiness JSON', id: 'JSON release readiness' }, file: 'stage2/s06_report/results/release_readiness.json', contains: { en: 'Gate verdict, violations, and summary metrics.', id: 'Gate verdict, violations, dan ringkasan metrik.' }, usedBy: { en: 'human review + automation', id: 'review manusia + otomasi' } },
        { name: { en: 'Final HTML report', id: 'Laporan HTML final' }, file: 'stage2/s06_report/results/mmlu_report.html', contains: { en: 'Narrative report with run outcomes.', id: 'Laporan naratif dengan outcome run.' }, usedBy: { en: 'presentation and handoff', id: 'presentasi dan handoff' } },
      ],
    },
  }
  return all[stageId]
}

export function ResidualStage() {
  const [lang, setLang] = useState<Lang>('en')
  const copy = useMemo(() => stageCopyFor('residual'), [])
  const { data, error, loading } = useFetchJson<ResidualProfile>(P.residualProfile)
  const meta = useFetchJson<{ group_counts?: Record<string, number>; n_rows?: number }>(P.residualMeta)

  const subjects = [...(data?.by_subject ?? [])].sort((a, b) => b.error_rate - a.error_rate).slice(0, 15)
  const groups = data?.group_counts ?? meta.data?.group_counts
  const maxG = groups ? Math.max(...Object.values(groups), 1) : 1

  return (
    <>
      <PipelineStepper activeId="residual" />
      <StageHeader
        step={t(copy.step, lang)}
        accent="#3B82F6"
        title={t(copy.title, lang)}
        description={t(copy.description, lang)}
        rightSlot={<StageLanguageToggle lang={lang} setLang={setLang} />}
      />
      <LoadState loading={loading || meta.loading} error={error ?? meta.error}>
        <GuidelineSections copy={copy} lang={lang} />
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-7 bg-surface-container-lowest rounded-xl p-6">
            <h3 className="font-headline font-bold mb-2">Error rate by subject</h3>
            <p className="text-sm text-on-surface-variant mb-4">
              Subject-level error rates reveal where failures are concentrated and provide context for residual-group interpretation.
            </p>
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
        <StageNavBlock activeId="residual" lang={lang} />
      </LoadState>
    </>
  )
}

export function ExportStage() {
  const [lang, setLang] = useState<Lang>('en')
  const copy = useMemo(() => stageCopyFor('export'), [])
  const { data, error, loading } = useFetchJson<ExportMeta>(P.exportMeta)
  return (
    <>
      <PipelineStepper activeId="export" />
      <StageHeader
        step={t(copy.step, lang)}
        accent="#6366F1"
        title={t(copy.title, lang)}
        description={t(copy.description, lang)}
        rightSlot={<StageLanguageToggle lang={lang} setLang={setLang} />}
      />
      <LoadState loading={loading} error={error}>
        <GuidelineSections copy={copy} lang={lang} />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-surface-container-lowest rounded-xl p-8 space-y-4">
            <h3 className="font-headline font-bold">Model & tensor</h3>
            <dl className="space-y-3 text-sm">
              <Row
                label="Model"
                value={data?.model}
                info="The Hugging Face model identifier used to generate hidden activations."
              />
              <Row
                label="Rows"
                value={data?.n_rows?.toLocaleString()}
                info="The number of dataset examples exported; row order matches the residual CSV input."
              />
              <Row
                label="Hidden dim"
                value={data?.hidden_dim?.toString()}
                info="The vector width of each pooled hidden state. For Gemma-2-9b, this is 3584 dimensions."
              />
              <Row
                label="Layer index"
                value={data?.layer_index?.toString()}
                info="Which transformer layer is read. -1 means the final layer (Python negative index convention)."
              />
            </dl>
          </div>
          <div className="bg-surface-container-low rounded-xl p-8 flex flex-col justify-center items-center text-center">
            <MaterialIcon name="dataset" className="text-5xl text-secondary mb-4" />
            <p className="text-on-surface-variant text-sm max-w-sm">
              Binary hidden tensor:{' '}
              <code className="text-xs bg-surface-container-lowest px-1 rounded">
                {data?.output_npz ?? 'mmlu_full_hidden.npz'}
              </code>{' '}
              (served under /pipeline_2026 for download or external tools).
            </p>
          </div>
        </div>
        <div className="mt-6 bg-surface-container-lowest rounded-xl p-6">
          <h3 className="font-headline font-bold mb-2">Hidden activation summary</h3>
          <p className="text-sm text-on-surface-variant mb-4">
            Visualization extracted from <code>mmlu_full_hidden.npz</code>: row-level norm distribution,
            per-dimension mean absolute activation, and cumulative PCA variance.
          </p>
          <img
            src={P.exportHiddenViz}
            alt="Summary visualization for hidden activation NPZ export"
            className="w-full rounded-lg bg-surface-container-low"
          />
        </div>
        <StageNavBlock activeId="export" lang={lang} />
      </LoadState>
    </>
  )
}

function Row({ label, value, info }: { label: string; value?: string; info?: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-outline-variant/10 pb-2">
      <dt className="text-on-surface-variant inline-flex items-center gap-2">
        <span>{label}</span>
        {info ? (
          <span className="relative inline-flex group">
            <button
              type="button"
              className="inline-flex items-center justify-center w-5 h-5 rounded-full text-on-surface-variant hover:text-on-surface hover:bg-surface-container-low cursor-help"
              aria-label={`${label} definition`}
            >
              <MaterialIcon name="info" className="text-sm leading-none" />
            </button>
            <span
              role="tooltip"
              className="pointer-events-none absolute z-20 left-1/2 -translate-x-1/2 top-7 w-64 rounded-md bg-surface-container-highest text-on-surface text-xs leading-snug p-2 shadow-lg border border-outline-variant/20 opacity-0 translate-y-1 transition-all duration-150 group-hover:opacity-100 group-hover:translate-y-0 group-focus-within:opacity-100 group-focus-within:translate-y-0"
            >
              {info}
            </span>
          </span>
        ) : null}
      </dt>
      <dd className="font-semibold text-right">{value ?? '—'}</dd>
    </div>
  )
}

export function SaeStage() {
  const [lang, setLang] = useState<Lang>('en')
  const copy = useMemo(() => stageCopyFor('sae'), [])
  const { data, error, loading } = useFetchJson<SaeMeta>(P.saeMeta)
  return (
    <>
      <PipelineStepper activeId="sae" />
      <StageHeader
        step={t(copy.step, lang)}
        accent="#8B5CF6"
        title={t(copy.title, lang)}
        description={t(copy.description, lang)}
        rightSlot={<StageLanguageToggle lang={lang} setLang={setLang} />}
      />
      <LoadState loading={loading} error={error}>
        <div className="space-y-6">
          <Section title={lang === 'en' ? 'What is happening?' : 'Apa yang sedang terjadi?'}>
            <p className="text-sm text-on-surface-variant mb-4">
              {lang === 'en'
                ? 'Dense hidden vectors are encoded into sparse latent features, then only the top informative latents are kept for group analysis.'
                : 'Vektor hidden yang padat di-encode menjadi fitur latent sparse, lalu hanya latent paling informatif yang dipakai untuk analisis grup.'}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm">
              {[
                { label: 'Dense vector (h)', color: '#6366F1' },
                { label: 'SAE encoder', color: '#8B5CF6' },
                { label: 'Sparse latent (z)', color: '#A855F7' },
                { label: 'Select top-k latents', color: '#EC4899' },
              ].map((node, i) => (
                <div key={node.label} className="flex items-center gap-2">
                  <div className="flex-1 rounded-lg border border-outline-variant/20 px-3 py-2 bg-surface-container-low">
                    <p className="font-semibold" style={{ color: node.color }}>
                      {node.label}
                    </p>
                  </div>
                  {i < 3 ? <MaterialIcon name="arrow_forward" className="text-on-surface-variant" /> : null}
                </div>
              ))}
            </div>
          </Section>

          <Section title={lang === 'en' ? 'What did we find?' : 'Apa yang ditemukan?'}>
            <p className="text-sm text-on-surface-variant mb-4">
              {lang === 'en'
                ? 'Most important features separating groups. Higher bars mean the latent is more useful for distinguishing error cases.'
                : 'Fitur terpenting untuk memisahkan grup. Batang yang lebih tinggi berarti latent lebih berguna untuk membedakan kasus error.'}
            </p>
            <img
              src={P.saeTopLatentsViz}
              alt="Top discriminative latent features"
              className="w-full rounded-lg bg-surface-container-low"
            />
          </Section>

          <Section title={lang === 'en' ? 'Why does it matter?' : 'Kenapa ini penting?'}>
            <p className="text-sm text-on-surface-variant mb-4">
              {lang === 'en'
                ? 'Simple High vs Low comparison per latent feature. This makes it easier to see which features activate differently across groups.'
                : 'Perbandingan sederhana High vs Low untuk setiap fitur latent. Ini memudahkan melihat fitur mana yang aktivasi-nya berbeda antar grup.'}
            </p>
            <img
              src={P.saeGroupCompareViz}
              alt="High vs low residual latent activation comparison"
              className="w-full rounded-lg bg-surface-container-low"
            />
          </Section>

          <Section title={lang === 'en' ? 'Can it separate data?' : 'Apakah bisa memisahkan data?'}>
            <p className="text-sm text-on-surface-variant mb-4">
              {lang === 'en'
                ? 'Intuition view: if the two colors form cleaner clusters, these latents separate groups more clearly.'
                : 'Tampilan intuisi: jika dua warna membentuk klaster yang lebih jelas, latent ini memisahkan grup dengan lebih baik.'}
            </p>
            <img
              src={P.saeGroupScatterViz}
              alt="High and low residual group separation scatter plot"
              className="w-full rounded-lg bg-surface-container-low"
            />
          </Section>

          <div className="bg-surface-container-lowest rounded-xl p-8 max-w-3xl">
            <h3 className="font-headline font-bold mb-4">
              {lang === 'en' ? 'SAE integrity snapshot' : 'Ringkasan integritas SAE'}
            </h3>
            <dl className="space-y-3 text-sm">
              <Row label="Combined model string" value={data?.model} />
              <Row label="Rows" value={data?.n_rows?.toLocaleString()} />
              <Row label="d_sae" value={data?.d_sae?.toLocaleString()} />
              <Row label="SAE release" value={data?.sae_release} />
              <Row label="SAE hook" value={data?.sae_id} />
            </dl>
            <p className="text-xs text-on-surface-variant mt-6">
              Output:{' '}
              <code className="bg-surface-container-low px-1 rounded">
                {data?.output_npz ?? 'mmlu_full_sae_latents.npz'}
              </code>
            </p>
          </div>
        </div>
        <StageNavBlock activeId="sae" lang={lang} />
      </LoadState>
    </>
  )
}

export function DiffStage() {
  const [lang, setLang] = useState<Lang>('en')
  const copy = useMemo(() => stageCopyFor('diff'), [])
  const diag = useFetchJson<DiffDiagnostics>(P.diffDiagnostics)
  const catalog = useFetchJson<PatternCatalog>(P.patternCatalog)
  const rows = (catalog.data?.patterns ?? []).slice(0, 25)

  return (
    <>
      <PipelineStepper activeId="diff" />
      <StageHeader
        step={t(copy.step, lang)}
        accent="#EC4899"
        title={t(copy.title, lang)}
        description={t(copy.description, lang)}
        rightSlot={<StageLanguageToggle lang={lang} setLang={setLang} />}
      />
      <LoadState loading={diag.loading || catalog.loading} error={diag.error ?? catalog.error}>
        <GuidelineSections copy={copy} lang={lang} />
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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <section className="bg-surface-container-lowest rounded-xl p-6">
            <h3 className="font-headline font-bold mb-2">Attribution-style significance histogram</h3>
            <p className="text-sm text-on-surface-variant mb-4">
              LessWrong-inspired view: distribution of pattern significance using <code>-log10(p-value)</code>.
              Farther right means stronger statistical evidence.
            </p>
            <img
              src={P.diffSignificanceHist}
              alt="Histogram of diff pattern significance"
              className="w-full rounded-lg bg-surface-container-low"
            />
          </section>
          <section className="bg-surface-container-lowest rounded-xl p-6">
            <h3 className="font-headline font-bold mb-2">Effect map of candidate latents</h3>
            <p className="text-sm text-on-surface-variant mb-4">
              Each point is one latent candidate. X-axis is <code>V&apos;</code> (effect size), Y-axis is
              significance; labels mark top latents by <code>V&apos;</code>.
            </p>
            <img
              src={P.diffEffectMap}
              alt="Scatter map of latent effect size versus significance"
              className="w-full rounded-lg bg-surface-container-low"
            />
          </section>
        </div>
        <StageNavBlock activeId="diff" lang={lang} />
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
  const [lang, setLang] = useState<Lang>('en')
  const copy = useMemo(() => stageCopyFor('eval'), [])
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
        step={t(copy.step, lang)}
        accent="#10B981"
        title={t(copy.title, lang)}
        description={t(copy.description, lang)}
        rightSlot={<StageLanguageToggle lang={lang} setLang={setLang} />}
      />
      <LoadState loading={loading} error={error}>
        <GuidelineSections copy={copy} lang={lang} />
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
        <StageNavBlock activeId="eval" lang={lang} />
      </LoadState>
    </>
  )
}

export function ReportStage() {
  const [lang, setLang] = useState<Lang>('en')
  const copy = useMemo(() => stageCopyFor('report'), [])
  const { data, error, loading } = useFetchJson<ReleaseReadiness>(P.releaseReadiness)
  const reportUrl = P.reportHtml

  return (
    <>
      <PipelineStepper activeId="report" />
      <StageHeader
        step={t(copy.step, lang)}
        accent="#0058BE"
        title={t(copy.title, lang)}
        description={t(copy.description, lang)}
        rightSlot={<StageLanguageToggle lang={lang} setLang={setLang} />}
      />
      <LoadState loading={loading} error={error}>
        <GuidelineSections copy={copy} lang={lang} />
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
        <StageNavBlock activeId="report" lang={lang} />
      </LoadState>
    </>
  )
}
