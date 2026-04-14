import { useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { MaterialIcon } from '../components/MaterialIcon'
import { PosterDiscoveredPattern } from '../components/PosterDiscoveredPattern'
import { PosterSetLevelMetrics } from '../components/PosterSetLevelMetrics'
import { ResidualDistributionChart } from '../components/ResidualDistributionChart'
import { SaeLatentDeltaHeatmap } from '../components/SaeLatentDeltaHeatmap'
import { LoadState } from '../components/LoadState'
import { P } from '../paths'
import { useFetchJson } from '../useFetchJson'
import type {
  EvalMetrics,
  PatternCatalog,
  ReleaseReadiness,
  ResidualProfile,
  ResidualHistogram,
  SubjectLatentDeltaHeatmap,
} from '../types'

function hasResidualHistogramBins(h: ResidualHistogram | undefined): boolean {
  if (!h) return false
  const n = (x: number[] | undefined) => x?.length ?? 0
  return n(h.counts) > 0 || n(h.zoom?.counts) > 0 || n(h.full?.counts) > 0
}

function formatPct(x: number | undefined, digits = 1) {
  if (x == null || Number.isNaN(x)) return '—'
  return `${(x * 100).toFixed(digits)}%`
}

function formatFixed(x: number | undefined, digits = 3) {
  if (x == null || Number.isNaN(x)) return '—'
  return x.toFixed(digits)
}

type PosterLanguage = 'en' | 'id'

const copy = {
  en: {
    back: 'Run selection',
    stageExplorer: 'Stage explorer',
    print: 'Print / Save PDF',
    badge: 'CS 6966/5966 · Interpretability & LLM errors · University of Utah',
    titleLine1: 'Improving LLM Failure Pattern Discovery',
    titleLine2: 'with Residual-Controlled and SAE-Diffing',
    authors: 'Ivan Andhika (u1590903) · Tushar Jain (U1528089)',
    affiliation: 'University of Utah',
    sectionProblem: '01 · The problem',
    problemHead: 'Why is automatic failure discovery hard?',
    problemQuote:
      'Large language models fail in systematic ways, yet those failures are difficult to characterize automatically—blocking trustworthy teaching pipelines.',
    problemBody:
      'An AI-integration teaching pipeline helps users recognize failure patterns, but Stage 2 (automatic discovery) remains the bottleneck. Prior work mixes prompting and embedding clustering with mixed results. We keep the teaching framework fixed and upgrade only how patterns are discovered.',
    sectionGaps: '02 · Limits of current practice',
    gap1: 'Evaluation often depends on predefined meta-labels (e.g., subject tags), so novel failures may be missed.',
    gap2: 'There is little agreement on how to compare whole pattern sets for consistency, overlap, and predictive usefulness.',
    gap3: 'Without difficulty control, “hard questions” can dominate and confound interpretable structure.',
    gap4: 'Most checks focus on single patterns or label overlap, not the quality of the full discovered set.',
    stage1CardTitle: 'Stage 1 · Difficulty residual',
    stage1CardBody:
      'Shallow features (length, numerals, simple structure) train a logistic model for expected error. Residual = actual − expected; we split high- vs low-residual cohorts before SAE analysis.',
    residualPngCap: 'Exported diagnostic: error rate by MMLU subject after residual labeling (full run).',
    stageDiffTitle: 'Stage 4 · Latent-level tests',
    stageDiffBody:
      'Histogram of significance across SAE dimensions (diff stage): many features are tested; top-ranked entries feed the pattern catalog.',
    diffHistCap: 'Distribution of latent-level test statistics (when this figure is present in the run).',
    gateTitle: 'Release gate',
    gateNote:
      'Thresholds encode pipeline QC (e.g., global error-rate bound), not whether the research question is answered. High raw error rate can be expected on this benchmark.',
    residualDistTitle: 'Residual distribution',
    residualDistCaption:
      'Histogram uses a zoomed x-range (p1–p99) so the shape is readable; the full residual span is wider and can collapse into one tall bin. Bar height uses √count. Teal/amber dashed lines are Q25/Q75 for cohort splits — on this run the IQR is very narrow, so the distribution looks sharply peaked even though the logistic fit is working as intended.',
    residualDistAxis: 'residual →',
    residualDistQ25: 'Q25 (low-residual cutoff)',
    residualDistQ75: 'Q75 (high-residual cutoff)',
    residualDistLoading: 'Loading histogram…',
    residualDistError: 'Histogram JSON missing — run build_residual_histogram_json.py.',
    methodsTitle: 'Methods — three-stage upgrade',
    pipe1Name: 'Residual control',
    pipe1Desc:
      'Predict expected error from shallow features; form high- and low-residual groups of mistakes.',
    pipe2Name: 'SAE diffing',
    pipe2Desc:
      'Encode layer-20 hidden states with a sparse autoencoder, then contrast activations between residual cohorts to surface candidate latents.',
    pipe3Name: 'Set-level evaluation',
    pipe3Desc:
      'Score entire pattern sets with coverage, concentration, predictive utility (AUC/F1), and redundancy (Jaccard), plus stability when multi-seed data exists.',
    statsTitle: 'End-to-end run · key numbers',
    resultsTitle: 'Results · figures from this repository',
    figSubject: 'Error rate by subject (top 10, this run)',
    subjectBarHint: 'Bar length = empirical error rate on held MMLU items. Longer bars mark subjects with more mistakes in this pass.',
    figHeatmap: 'SAE latent Δ heatmap (high − low residual)',
    figHeatmapCap:
      'Rows = MMLU subjects (ranked by High+Low residual count). Columns = top-12 catalog latents. Cell = mean SAE activation on High residual minus mean on Low residual for that subject. Gray = too few rows per cohort (see min n). Hover for exact Δ and cohort sizes.',
    heatmapLegendLow: 'Low Δ',
    heatmapLegendHigh: 'High Δ',
    heatmapNull: 'Insufficient n per cohort',
    heatmapLoading: 'Loading heatmap…',
    heatmapError: 'Heatmap JSON missing — run build_subject_latent_heatmap.py on this machine.',
    figEffect: 'Latent effect map (high − low residual)',
    figEffectCap:
      'Stage-4 visualization of where SAE features differ most between cohorts (export from diff stage; read as exploratory, not causal proof).',
    figSae: 'SAE latent structure (research viz)',
    figSaeCap: 'Stage-3 encoding summary: how sparse latents organize across the merged MMLU table for this checkpoint.',
    figEval: 'Residual vs SAE evaluation summary',
    figEvalCap: 'Stage-5 bar summary comparing residual-defined groups and SAE-based pattern coverage on this Gemma run.',
    figExportTag: 'Export',
    figExportTitle: 'Hidden-state sample (layer export)',
    figExportCap:
      'Stage-2 sanity check: projected hidden states used before SAE encoding (when this asset exists in the run).',
    patternDisclaimer:
      'Interpretation is hypothesis-level: the pipeline ranks latents by cohort contrast; automatic semantic labels are not asserted here.',
    discPatternFig: 'Discovered pattern — example',
    discPatternBody:
      'Top-ranked catalog entry from Welch-style contrast on High vs Low residual SAE activations. The pipeline does not assign semantic names; this is a concrete candidate latent for inspection.',
    discPatternCaption:
      'Group comparison figure (SAE activations): high- vs low-residual cohorts on this run (when the PNG is present).',
    discPatternEmpty: 'Pattern catalog not loaded.',
    setLevelFig: 'Set-level metrics & stability',
    setLevelCaption:
      'Radar axes are on 0–1 scale. Distinctness = 1 − avg pairwise Jaccard (higher = less overlap between patterns). Amber dashed ring = 0.5 on every axis (visual guide, not a tuned baseline).',
    setLevelRadarHelp:
      'Teal polygon = this run. Larger area usually means stronger coverage/concentration/PU and more distinct patterns; AUC near 0.5 still indicates weak ranking quality.',
    setLevelTable: 'Set-level summary',
    setLevelAxisCov: 'Cov',
    setLevelAxisConc: 'Conc',
    setLevelAxisAuc: 'AUC',
    setLevelAxisF1: 'F1',
    setLevelAxisDistinct: 'Distinct',
    setLevelLegendOur: 'This run',
    setLevelLegendRef: '0.5 reference ring',
    setLevelStabilityHeading: 'Stability across seeds',
    setLevelStabilityNoData:
      'Not computed in this export (single split / pairs_evaluated=0). Re-run evaluation with multi-seed overlap to populate overlap@k.',
    metricsExplain: 'Reading the metrics',
    metricCov: 'Pattern coverage',
    metricCovNote: 'share of errors touched by the union of patterns (set-level)',
    metricConc: 'Concentration',
    metricConcNote: 'error density within flagged instances',
    metricAuc: 'AUC (PU)',
    metricAucNote: 'ranking quality of pattern-based detector',
    metricF1: 'F1 (PU)',
    metricF1Note: 'can look non-trivial under imbalance',
    metricJac: 'Avg Jaccard',
    metricJacNote: 'pairwise overlap between patterns',
    metricStab: 'Stability',
    metricStabNote: 'multi-seed overlap at top-k (if computed)',
    coreTitle: '03 · Core finding',
    coreQuote:
      'After difficulty control, a substantial share of errors falls in the high-residual cohort; SAE diffing yields a large candidate pattern set with measurable redundancy and near-random AUC—useful for iteration, not final verdict.',
    storyTitle: 'The story in the numbers',
    limitsTitle: '04 · Limitations & next steps',
    lim1:
      'Peer feedback: predictions and hidden states should ideally come from one aligned setup; we note the risk of surrogate alignment and plan tighter open-weight end-to-end checks.',
    lim2: 'Runtime and compute (full MMLU + SAE) limit how many seeds and ablations we can run locally; heavier jobs target CHPC.',
    lim3: 'Predictive utility (AUC ≈ 0.5 here) implies latent membership alone is weak; deduplication and richer features remain open work.',
    conclusions: 'Conclusions',
    c1: 'The residual → SAE → set-metric pipeline runs end-to-end on Gemma 2 9B + MMLU with reproducible artifacts in this repo.',
    c2: 'Difficulty-aware residuals isolate “surprising” errors; coverage of errors in the high-residual group is a concrete, reportable signal.',
    c3: 'Set-level metrics make pattern-set quality discussable (coverage, concentration, redundancy), even when raw predictive power is modest.',
    c4: 'Future work: prompting or embedding baselines, ablations, and stability across more seeds on matched checkpoints.',
    references: 'References',
    acknowledgments: 'Acknowledgments',
    ackBody:
      'CS 6966/5966 instructors and peers; University of Utah CHPC for GPU time; SAELens / Hugging Face tooling; shared codebase contributors.',
    ref1:
      'Stringham, N., Chaleshtori, F. H., Yan, X., Xu, Z., Wang, B., & Marasović, A. (2025). Teaching people LLM’s errors and getting it right. arXiv:2512.21422.',
    ref2:
      'Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. arXiv:2309.08600.',
    ref3:
      'Jiang, N., Sun, X., Dunlap, L., Smith, L., & Nanda, N. (2025). Interpretable embeddings with sparse autoencoders: A data analysis toolkit. arXiv:2512.10092.',
    behavioralLabel: 'Behavioral model',
    saeLabel: 'SAE',
    benchmarkLabel: 'Benchmark',
    dimsChip: 'Dimensions',
    errorRate: 'Error rate',
    highResidualCov: 'High-res. coverage',
    patternsLabel: 'Patterns',
  },
  id: {
    back: 'Pilih run',
    stageExplorer: 'Explorer stage',
    print: 'Cetak / Simpan PDF',
    badge: 'CS 6966/5966 · Interpretability & kesalahan LLM · University of Utah',
    titleLine1: 'Meningkatkan Penemuan Pola Kegagalan LLM',
    titleLine2: 'dengan Residual-Terkontrol dan SAE-Diffing',
    authors: 'Ivan Andhika (u1590903) · Tushar Jain (U1528089)',
    affiliation: 'University of Utah',
    sectionProblem: '01 · Masalah',
    problemHead: 'Mengapa penemuan pola otomatis sulit?',
    problemQuote:
      'Model bahasa besar gagal secara sistematis, tetapi pola kegagalannya sulit dikarakterisasi secara otomatis—menghambat pipeline pengajaran yang andal.',
    problemBody:
      'Pipeline pengajaran integrasi AI membantu pengguna mengenali pola kegagalan, tetapi Stage 2 (penemuan otomatis) masih menjadi bottleneck. Studi sebelumnya mencampur prompting dan clustering embedding dengan hasil campuran. Kami mempertahankan kerangka pengajaran dan hanya meningkatkan cara pola ditemukan.',
    sectionGaps: '02 · Batas praktik saat ini',
    gap1: 'Evaluasi sering bergantung pada meta-label (mis. tag subject), sehingga kegagalan baru bisa terlewat.',
    gap2: 'Kurang ada kesepakatan cara membandingkan keseluruhan set pola untuk konsistensi, tumpang-tindih, dan kegunaan prediktif.',
    gap3: 'Tanpa kontrol kesulitan, soal yang sulit bisa mendominasi dan mengaburkan struktur yang dapat diinterpretasi.',
    gap4: 'Banyak pemeriksaan fokus pada pola tunggal atau overlap label, bukan kualitas set penuh yang ditemukan.',
    stage1CardTitle: 'Stage 1 · Residual kesulitan',
    stage1CardBody:
      'Fitur dangkal (panjang, angka, struktur sederhana) melatih model logistik untuk error yang diharapkan. Residual = aktual − ekspektasi; kami memisahkan kohort residual tinggi vs rendah sebelum analisis SAE.',
    residualPngCap: 'Diagnostik ekspor: error rate per subject MMLU setelah pelabelan residual (run penuh).',
    stageDiffTitle: 'Stage 4 · Uji level latent',
    stageDiffBody:
      'Histogram signifikansi lintas dimensi SAE (tahap diff): banyak fitur diuji; entri peringkat teratas mengisi katalog pola.',
    diffHistCap: 'Distribusi statistik uji level latent (jika gambar ini ada pada run).',
    gateTitle: 'Gate rilis',
    gateNote:
      'Ambang batas adalah QC pipeline (mis. batas error rate global), bukan apakah pertanyaan riset terjawab. Error rate mentah tinggi bisa wajar pada benchmark ini.',
    residualDistTitle: 'Distribusi residual',
    residualDistCaption:
      'Histogram memakai rentang-x zoom (p1–p99) agar bentuknya terbaca; rentang residual penuh lebih lebar dan bisa menyatu jadi satu batang tinggi. Tinggi batang memakai √count. Garis putus teal/amber = Q25/Q75 untuk split kohort — pada run ini IQR sangat sempit, jadi distribusi terlihat sangat runcing walau model logistic tetap konsisten.',
    residualDistAxis: 'residual →',
    residualDistQ25: 'Q25 (batas low-residual)',
    residualDistQ75: 'Q75 (batas high-residual)',
    residualDistLoading: 'Memuat histogram…',
    residualDistError: 'JSON histogram tidak ada — jalankan build_residual_histogram_json.py.',
    methodsTitle: 'Metode — tiga tahap peningkatan',
    pipe1Name: 'Kontrol residual',
    pipe1Desc:
      'Prediksi error yang diharapkan dari fitur dangkal; bentuk grup salah residual tinggi dan rendah.',
    pipe2Name: 'SAE diffing',
    pipe2Desc:
      'Encode hidden state layer-20 dengan sparse autoencoder, lalu bandingkan aktivasi antar kohort residual untuk kandidat latent.',
    pipe3Name: 'Evaluasi set-level',
    pipe3Desc:
      'Skor seluruh set pola dengan coverage, concentration, predictive utility (AUC/F1), dan redundansi (Jaccard), plus stabilitas jika ada multi-seed.',
    statsTitle: 'Run end-to-end · angka kunci',
    resultsTitle: 'Hasil · gambar dari repositori ini',
    figSubject: 'Error rate per subject (10 teratas, run ini)',
    subjectBarHint: 'Panjang bar = error rate empiris pada item MMLU. Bar lebih panjang = lebih banyak kesalahan pada subject tersebut.',
    figHeatmap: 'Heatmap Δ latent SAE (residual tinggi − rendah)',
    figHeatmapCap:
      'Baris = subject MMLU (diurutkan jumlah baris High+Low residual). Kolom = 12 latent teratas dari katalog. Sel = rata aktivasi SAE pada kohort High dikurangi Low untuk subject itu. Abu-abu = baris terlalu sedikit per kohort. Arahkan kursor untuk Δ dan ukuran kohort.',
    heatmapLegendLow: 'Δ rendah',
    heatmapLegendHigh: 'Δ tinggi',
    heatmapNull: 'n per kohort tidak cukup',
    heatmapLoading: 'Memuat heatmap…',
    heatmapError: 'JSON heatmap tidak ada — jalankan build_subject_latent_heatmap.py.',
    figEffect: 'Peta efek latent (residual tinggi − rendah)',
    figEffectCap:
      'Visualisasi Stage-4 perbedaan fitur SAE antar kohort (ekspor tahap diff; baca sebagai eksplorasi, bukti kausal).',
    figSae: 'Struktur latent SAE (viz riset)',
    figSaeCap: 'Ringkasan Stage-3 encoding: bagaimana latent sparse mengorganisir tabel MMLU untuk checkpoint ini.',
    figEval: 'Ringkasan evaluasi residual vs SAE',
    figEvalCap: 'Ringkasan bar Stage-5 untuk grup residual dan cakupan pola berbasis SAE pada run Gemma ini.',
    figExportTag: 'Ekspor',
    figExportTitle: 'Sampel hidden state (layer export)',
    figExportCap:
      'Sanity check Stage-2: proyeksi hidden state sebelum encoding SAE (jika aset ini ada pada run).',
    patternDisclaimer:
      'Ini level hipotesis: pipeline meranking latent menurut kontras kohort; label semantik otomatis tidak diklaim di sini.',
    discPatternFig: 'Pola terdiskover — contoh',
    discPatternBody:
      'Entri katalog peringkat teratas dari kontras Welch pada aktivasi SAE kohort residual Tinggi vs Rendah. Pipeline tidak memberi nama semantik; ini kandidat latent konkret untuk inspeksi.',
    discPatternCaption:
      'Gambar perbandingan grup (aktivasi SAE): kohort residual tinggi vs rendah pada run ini (jika PNG ada).',
    discPatternEmpty: 'Katalog pola belum dimuat.',
    setLevelFig: 'Metrik set-level & stabilitas',
    setLevelCaption:
      'Sumbu radar pada skala 0–1. Distinctness = 1 − rata Jaccard berpasangan (lebih tinggi = lebih sedikit overlap antarpola). Ring putus amber = 0.5 pada tiap sumbu (panduan visual, bukan baseline terselaraskan).',
    setLevelRadarHelp:
      'Poligon teal = run ini. Area lebih besar biasanya berarti coverage/konsentrasi/PU lebih kuat dan pola lebih terpisah; AUC ~0.5 tetap menandakan ranking lemah.',
    setLevelTable: 'Ringkasan set-level',
    setLevelAxisCov: 'Cov',
    setLevelAxisConc: 'Conc',
    setLevelAxisAuc: 'AUC',
    setLevelAxisF1: 'F1',
    setLevelAxisDistinct: 'Distinct',
    setLevelLegendOur: 'Run ini',
    setLevelLegendRef: 'Ring referensi 0.5',
    setLevelStabilityHeading: 'Stabilitas lintas seed',
    setLevelStabilityNoData:
      'Belum dihitung pada ekspor ini (satu split / pairs_evaluated=0). Jalankan eval dengan overlap multi-seed untuk mengisi overlap@k.',
    metricsExplain: 'Cara membaca metrik',
    metricCov: 'Coverage pola',
    metricCovNote: 'porsi error yang tersentuh oleh gabungan pola (set-level)',
    metricConc: 'Concentration',
    metricConcNote: 'kepadatan error pada instance yang ditandai',
    metricAuc: 'AUC (PU)',
    metricAucNote: 'kualitas ranking detektor berbasis pola',
    metricF1: 'F1 (PU)',
    metricF1Note: 'bisa tampil menarik saat imbalance',
    metricJac: 'Avg Jaccard',
    metricJacNote: 'overlap berpasangan antarpola',
    metricStab: 'Stability',
    metricStabNote: 'overlap multi-seed pada top-k (jika dihitung)',
    coreTitle: '03 · Temuan inti',
    coreQuote:
      'Setelah kontrol kesulitan, bagian besar error jatuh pada kohort residual tinggi; SAE diffing menghasilkan banyak kandidat pola dengan redundansi terukur dan AUC mendekati acak—berguna untuk iterasi, bukan putusan akhir.',
    storyTitle: 'Alur angka',
    limitsTitle: '04 · Keterbatasan & langkah lanjut',
    lim1:
      'Masukan reviewer: prediksi dan hidden state idealnya dari satu setup yang selaras; kami catat risiko alignment surrogate dan rencana pengecekan open-weight yang lebih ketat.',
    lim2: 'Runtime dan compute (MMLU penuh + SAE) membatasi jumlah seed dan ablasi lokal; job berat ditargetkan ke CHPC.',
    lim3: 'Predictive utility (AUC ≈ 0.5 di sini) berarti membership latent saja masih lemah; deduplikasi dan fitur kaya masih pekerjaan terbuka.',
    conclusions: 'Kesimpulan',
    c1: 'Pipeline residual → SAE → metrik set-level berjalan end-to-end pada Gemma 2 9B + MMLU dengan artefak yang dapat direproduksi di repo ini.',
    c2: 'Residual yang sadar kesulitan mengisolasi error yang “mengejutkan”; coverage error di grup residual tinggi adalah sinyal konkret.',
    c3: 'Metrik set-level membuat kualitas set pola dapat dibahas (coverage, concentration, redundansi), walau daya prediksi mentah sederhana.',
    c4: 'Kerja lanjut: baseline prompting/embedding, ablasi, dan stabilitas lintas lebih banyak seed pada checkpoint yang selaras.',
    references: 'Referensi',
    acknowledgments: 'Ucapan terima kasih',
    ackBody:
      'Instruktur dan rekan CS 6966/5966; CHPC University of Utah untuk GPU; toolchain SAELens / Hugging Face; kontributor codebase bersama.',
    ref1:
      'Stringham, N., Chaleshtori, F. H., Yan, X., Xu, Z., Wang, B., & Marasović, A. (2025). Teaching people LLM’s errors and getting it right. arXiv:2512.21422.',
    ref2:
      'Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. arXiv:2309.08600.',
    ref3:
      'Jiang, N., Sun, X., Dunlap, L., Smith, L., & Nanda, N. (2025). Interpretable embeddings with sparse autoencoders: A data analysis toolkit. arXiv:2512.10092.',
    behavioralLabel: 'Model perilaku',
    saeLabel: 'SAE',
    benchmarkLabel: 'Benchmark',
    dimsChip: 'Dimensi',
    errorRate: 'Tingkat error',
    highResidualCov: 'Coverage residual tinggi',
    patternsLabel: 'Pola',
  },
} as const

function SubjectBarChart({
  rows,
  title,
  hint,
}: {
  rows: { label: string; error_rate: number; n: number }[]
  title: string
  hint: string
}) {
  const max = Math.max(...rows.map((r) => r.error_rate), 1e-6)
  return (
    <figure className="rounded-md border border-slate-200 bg-white p-2.5 shadow-sm">
      <p className="font-headline text-[11px] font-bold text-slate-900 mb-0.5 leading-tight">{title}</p>
      <p className="text-[9px] text-slate-600 mb-2">{hint}</p>
      <div className="space-y-1">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2 text-[9px]">
            <span className="w-[6.5rem] shrink-0 truncate text-right text-slate-800 font-medium" title={r.label}>
              {r.label.replace(/_/g, ' ')}
            </span>
            <div className="flex-1 h-3.5 bg-slate-100 rounded overflow-hidden border border-slate-100">
              <div
                className="h-full rounded bg-gradient-to-r from-amber-500 to-amber-600"
                style={{ width: `${(r.error_rate / max) * 100}%` }}
              />
            </div>
            <span className="w-9 shrink-0 tabular-nums text-slate-600">{formatPct(r.error_rate, 0)}</span>
          </div>
        ))}
      </div>
    </figure>
  )
}

function OptionalFigure({ src, alt }: { src: string; alt: string }) {
  const [ok, setOk] = useState(true)
  if (!ok) return null
  return (
    <img
      src={src}
      alt={alt}
      className="w-full rounded border border-slate-200 bg-white"
      onError={() => setOk(false)}
    />
  )
}

function PosterCard({
  children,
  className = '',
  accent = 'amber',
}: {
  children: ReactNode
  className?: string
  accent?: 'amber' | 'teal' | 'none'
}) {
  const top =
    accent === 'amber'
      ? 'border-t-2 border-t-amber-500'
      : accent === 'teal'
        ? 'border-t-2 border-t-teal-600'
        : ''
  return (
    <div
      className={`rounded-md border border-slate-200 bg-white p-3 shadow-sm ${top} ${className}`}
    >
      {children}
    </div>
  )
}

function Chip({ children }: { children: ReactNode }) {
  return (
    <span className="inline-block font-mono text-[8px] uppercase tracking-[0.14em] text-amber-900 bg-amber-100 border border-amber-200/80 px-2 py-0.5 rounded">
      {children}
    </span>
  )
}

function FigCard({
  tag,
  tagAccent,
  title,
  children,
  caption,
}: {
  tag: string
  tagAccent?: boolean
  title: string
  children: ReactNode
  caption: string
}) {
  return (
    <figure className="rounded-md border border-slate-200 bg-white overflow-hidden shadow-sm">
      <div className="flex items-baseline gap-2 border-b border-slate-200 px-2.5 py-2 bg-slate-50/80">
        <span
          className={`font-mono text-[8px] uppercase tracking-wider ${tagAccent ? 'text-amber-700' : 'text-teal-700'}`}
        >
          {tag}
        </span>
        <span className="text-[11px] font-semibold text-slate-900">{title}</span>
      </div>
      <div className="p-2.5">
        {children}
        <figcaption className="text-[9px] text-slate-600 mt-2 italic leading-snug">{caption}</figcaption>
      </div>
    </figure>
  )
}

export function Poster() {
  return <PosterPage lang="en" />
}

export function PosterEN() {
  return <PosterPage lang="en" />
}

export function PosterID() {
  return <PosterPage lang="id" />
}

function PosterPage({ lang }: { lang: PosterLanguage }) {
  const t = copy[lang]
  const rr = useFetchJson<ReleaseReadiness>(P.releaseReadiness)
  const profile = useFetchJson<ResidualProfile>(P.mmluFullResidualProfile)
  const evFallback = useFetchJson<EvalMetrics>(P.evalMetrics)
  const patternCatalog = useFetchJson<PatternCatalog>(P.patternCatalog)
  const heatmapJson = useFetchJson<SubjectLatentDeltaHeatmap>(P.subjectLatentDeltaHeatmap)
  const residualHistJson = useFetchJson<ResidualHistogram>(P.residualHistogram)

  const loading = rr.loading || profile.loading || patternCatalog.loading
  const error = rr.error ?? profile.error ?? patternCatalog.error

  const topSubjects = useMemo(() => {
    const ps = rr.data?.per_subject
    if (ps?.length) {
      return [...ps]
        .sort((a, b) => b.error_rate - a.error_rate)
        .slice(0, 10)
        .map((s) => ({
          label: s.subject,
          error_rate: s.error_rate,
          n: s.n,
        }))
    }
    const by = profile.data?.by_subject
    if (by?.length) {
      return [...by]
        .sort((a, b) => b.error_rate - a.error_rate)
        .slice(0, 10)
        .map((s) => ({
          label: s.subject,
          error_rate: s.error_rate,
          n: s.n,
        }))
    }
    return []
  }, [rr.data?.per_subject, profile.data?.by_subject])

  const overall = rr.data?.overall
  const repr = rr.data?.representation_model
  const behavioral = rr.data?.behavioral_model?.mode ?? '—'
  const fin = rr.data?.final_set_level_metrics
  const resEval = rr.data?.residual_metrics_from_eval
  const evPU = fin?.predictive_utility ?? evFallback.data?.set_level_metrics?.predictive_utility
  const evRed = fin?.redundancy ?? evFallback.data?.set_level_metrics?.redundancy
  const nPatterns = fin?.n_patterns ?? evFallback.data?.set_level_metrics?.n_patterns
  const highCov = resEval?.high_residual?.coverage_of_errors ?? evFallback.data?.residual_metrics?.high_residual?.coverage_of_errors
  const highN = resEval?.high_residual?.n
  const covSet = fin?.coverage_concentration?.coverage
  const conc = fin?.coverage_concentration?.concentration
  const stab = fin?.stability
  const topPattern = useMemo(() => patternCatalog.data?.patterns?.[0], [patternCatalog.data?.patterns])

  const dSae = repr?.d_sae ?? 16384

  const stabilityDetailLine = useMemo(() => {
    const s = fin?.stability
    if (
      s?.pairs_evaluated &&
      s.pairs_evaluated > 0 &&
      s.overlap_at_k_mean != null &&
      Number.isFinite(s.overlap_at_k_mean)
    ) {
      return lang === 'en'
        ? `overlap@k=${formatFixed(s.overlap_at_k_mean)} · pairs=${s.pairs_evaluated} · top_k=${s.top_k ?? '—'}`
        : `overlap@k=${formatFixed(s.overlap_at_k_mean)} · pasangan=${s.pairs_evaluated} · top_k=${s.top_k ?? '—'}`
    }
    return null
  }, [fin?.stability, lang])

  const discPatternHeadline = useMemo(() => {
    if (!topPattern) return ''
    const rank = (topPattern.pattern_id ?? 0) + 1
    const sign = topPattern.V_prime >= 0 ? '+' : ''
    const vp = `${sign}${formatFixed(topPattern.V_prime, 4)}`
    if (lang === 'id') {
      return `Dim latent SAE #${topPattern.latent_id} · V′ = ${vp} · peringkat #${rank} dari ${dSae.toLocaleString()}`
    }
    return `SAE latent dim #${topPattern.latent_id} · V′ = ${vp} · rank #${rank} of ${dSae.toLocaleString()}`
  }, [topPattern, dSae, lang])

  const discPatternTags = useMemo(() => {
    if (!topPattern) return [] as { label: string }[]
    const cov = formatPct(covSet, 1)
    return [
      { label: `#${topPattern.latent_id}` },
      { label: `V′=${formatFixed(topPattern.V_prime, 4)}` },
      { label: `p=${topPattern.p_value.toExponential(2)}` },
      { label: lang === 'id' ? `cakupan≈${cov}` : `coverage≈${cov}` },
    ]
  }, [topPattern, covSet, lang])

  const nSubjects = rr.data?.per_subject?.length ?? 57
  const nInst = overall?.n_instances ?? repr?.n_rows ?? 7851

  const stabilityDisplay =
    stab?.overlap_at_k_mean != null && Number.isFinite(stab.overlap_at_k_mean)
      ? formatFixed(stab.overlap_at_k_mean, 3)
      : '—'

  const interpJaccard =
    lang === 'en'
      ? `Avg Jaccard ${formatFixed(evRed?.avg_pairwise_jaccard)} indicates overlap between discovered patterns (higher → more redundancy).`
      : `Avg Jaccard ${formatFixed(evRed?.avg_pairwise_jaccard)} menunjukkan overlap antarpola (lebih tinggi → lebih redundan).`

  const interpAuc =
    lang === 'en'
      ? `AUC ${formatFixed(evPU?.auc)} is close to random ranking for PU scoring—treat as a baseline to improve, not failure of the whole approach.`
      : `AUC ${formatFixed(evPU?.auc)} mendekati ranking acak untuk skor PU—anggap baseline untuk ditingkatkan, bukan kegagalan total pendekatan.`

  const gateVerdict = rr.data?.gates?.verdict
  const gateDetail = rr.data?.gates?.violations?.[0]?.detail

  return (
    <div className="min-h-screen bg-[#e8ecf2] text-slate-800 font-body poster-page-root">
      <div className="poster-screen-only sticky top-0 z-40 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white/95 px-4 py-2 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <Link to="/" className="inline-flex items-center gap-1 text-sm font-semibold text-teal-700 hover:underline">
            <MaterialIcon name="arrow_back" className="text-base" />
            {t.back}
          </Link>
          <Link to="/stage/residual" className="text-sm text-slate-600 hover:text-slate-900">
            {t.stageExplorer}
          </Link>
        </div>
        <button
          type="button"
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-bold text-white hover:bg-slate-800"
        >
          <MaterialIcon name="print" className="text-base" />
          {t.print}
        </button>
      </div>

      <div className="poster-print-scope flex justify-center px-2 pb-10 pt-3 print:pb-0 print:pt-0">
        <LoadState loading={loading} error={error}>
          <article
            className="poster-canvas w-full max-w-[min(96vw,1120px)] bg-white shadow-float print:shadow-none print:max-w-none rounded-lg border border-slate-200 overflow-hidden print:rounded-none print:border-0"
            style={{ aspectRatio: 'auto' }}
          >
            <div className="min-h-0 flex flex-col text-[11px] leading-snug p-4 md:p-5 gap-3 text-slate-800">
              {/* Header — template hdr */}
              <header className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-4 border-b border-slate-200 pb-4">
                <div>
                  <div className="mb-2">
                    <Chip>{t.badge}</Chip>
                  </div>
                  <h1 className="font-headline text-xl md:text-2xl font-extrabold text-slate-900 leading-tight tracking-tight">
                    {t.titleLine1}
                    <br />
                    <span className="text-amber-800">{t.titleLine2}</span>
                  </h1>
                  <p className="mt-2 text-[12px] font-semibold text-slate-700">{t.authors}</p>
                  <p className="text-[10px] text-slate-600">{t.affiliation}</p>
                </div>
                <div className="text-left lg:text-right font-mono text-[9px] text-slate-600 leading-relaxed space-y-1">
                  <p>
                    <span className="text-slate-500">{t.behavioralLabel}</span>{' '}
                    <code className="text-slate-800 bg-slate-100 px-1 rounded">{behavioral}</code>
                  </p>
                  <p>
                    <span className="text-slate-500">{t.saeLabel}</span>{' '}
                    <code className="text-slate-800 bg-slate-100 px-1 rounded text-[8px] break-all">
                      {repr?.sae_release ?? 'gemma-scope-9b-pt-res'} · {repr?.sae_id ?? 'layer_20/width_16k'}
                    </code>
                  </p>
                  <p>
                    <span className="text-slate-500">{t.benchmarkLabel}</span> MMLU · {nSubjects} subjects ·{' '}
                    {nInst.toLocaleString()} instances
                  </p>
                  {repr?.d_in != null && repr?.d_sae != null ? (
                    <p>
                      <span className="inline-block mt-1 bg-teal-100 text-teal-900 border border-teal-200/80 px-2 py-0.5 rounded text-[8px]">
                        {t.dimsChip}: d_in={repr.d_in} · d_sae={repr.d_sae}
                      </span>
                    </p>
                  ) : null}
                </div>
              </header>

              <div className="grid grid-cols-1 lg:grid-cols-[minmax(200px,240px)_1fr_minmax(200px,240px)] gap-3 items-start">
                {/* LEFT */}
                <div className="flex flex-col gap-3">
                  <PosterCard accent="amber">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-amber-800 mb-1.5">{t.sectionProblem}</p>
                    <h2 className="font-headline text-[13px] font-bold text-slate-900 mb-1">{t.problemHead}</h2>
                    <p className="text-[10px] text-slate-600 border-l-2 border-amber-400 pl-2 italic my-2 leading-relaxed">
                      {t.problemQuote}
                    </p>
                    <p className="text-[10px] text-slate-600 leading-relaxed">{t.problemBody}</p>
                  </PosterCard>

                  <PosterCard>
                    <p className="font-mono text-[9px] uppercase tracking-widest text-amber-800 mb-1.5">{t.sectionGaps}</p>
                    <ul className="space-y-2 text-[10px] text-slate-600">
                      <li className="flex gap-2">
                        <span className="text-amber-700 font-bold shrink-0">✗</span>
                        <span>{t.gap1}</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-amber-700 font-bold shrink-0">✗</span>
                        <span>{t.gap2}</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-amber-700 font-bold shrink-0">✗</span>
                        <span>{t.gap3}</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-amber-700 font-bold shrink-0">✗</span>
                        <span>{t.gap4}</span>
                      </li>
                    </ul>
                  </PosterCard>

                  <PosterCard accent="none" className="p-2.5">
                    {residualHistJson.loading ? (
                      <p className="text-[10px] text-slate-500">{t.residualDistLoading}</p>
                    ) : residualHistJson.error || !hasResidualHistogramBins(residualHistJson.data) ? (
                      <p className="text-[10px] text-slate-500">{t.residualDistError}</p>
                    ) : (
                      <ResidualDistributionChart
                        data={residualHistJson.data}
                        title={t.residualDistTitle}
                        caption={t.residualDistCaption}
                        axisLabel={t.residualDistAxis}
                        labelQ25={t.residualDistQ25}
                        labelQ75={t.residualDistQ75}
                      />
                    )}
                  </PosterCard>

                  <PosterCard>
                    <p className="font-mono text-[9px] uppercase tracking-widest text-slate-600 mb-1">{t.gateTitle}</p>
                    {gateVerdict ? (
                      <div
                        className={`inline-flex items-center gap-1.5 font-mono text-[9px] px-2 py-1 rounded-full border mb-2 ${
                          gateVerdict === 'pass'
                            ? 'bg-teal-50 text-teal-800 border-teal-200'
                            : 'bg-red-50 text-red-800 border-red-200'
                        }`}
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-current" />
                        {gateVerdict.toUpperCase()}
                        {gateDetail ? ` — ${gateDetail}` : ''}
                      </div>
                    ) : (
                      <p className="text-[9px] text-slate-500 mb-2">—</p>
                    )}
                    <p className="text-[10px] text-slate-600 leading-relaxed">{t.gateNote}</p>
                  </PosterCard>
                </div>

                {/* CENTER */}
                <div className="flex flex-col gap-3 min-w-0">
                  <div className="rounded-md border border-slate-200 bg-slate-50/90 p-3 shadow-sm border-t-2 border-t-amber-500">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-amber-800 mb-2">{t.methodsTitle}</p>
                    <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr_auto_1fr] gap-1 items-stretch">
                      <div className="rounded border border-slate-200 bg-white p-2">
                        <p className="font-mono text-lg font-bold text-amber-700 leading-none mb-1">01</p>
                        <p className="text-[11px] font-semibold text-slate-900">{t.pipe1Name}</p>
                        <p className="text-[9.5px] text-slate-600 mt-1 leading-relaxed">{t.pipe1Desc}</p>
                      </div>
                      <div className="hidden sm:flex items-center justify-center text-amber-600 text-lg">→</div>
                      <div className="rounded border border-slate-200 bg-white p-2">
                        <p className="font-mono text-lg font-bold text-teal-700 leading-none mb-1">02</p>
                        <p className="text-[11px] font-semibold text-slate-900">{t.pipe2Name}</p>
                        <p className="text-[9.5px] text-slate-600 mt-1 leading-relaxed">{t.pipe2Desc}</p>
                      </div>
                      <div className="hidden sm:flex items-center justify-center text-amber-600 text-lg">→</div>
                      <div className="rounded border border-slate-200 bg-white p-2">
                        <p className="font-mono text-lg font-bold text-slate-800 leading-none mb-1 bg-gradient-to-r from-amber-600 to-teal-600 bg-clip-text text-transparent">
                          03
                        </p>
                        <p className="text-[11px] font-semibold text-slate-900">{t.pipe3Name}</p>
                        <p className="text-[9.5px] text-slate-600 mt-1 leading-relaxed">{t.pipe3Desc}</p>
                      </div>
                    </div>
                  </div>

                  <PosterCard accent="none" className="bg-slate-50/50">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-amber-800 mb-2">{t.statsTitle}</p>
                    <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                      <div className="rounded border border-slate-200 bg-white p-2 text-center">
                        <span className="font-mono text-base font-bold text-amber-700 block tabular-nums">
                          {nInst.toLocaleString()}
                        </span>
                        <span className="text-[8px] uppercase text-slate-500 tracking-wide">{t.benchmarkLabel}</span>
                      </div>
                      <div className="rounded border border-slate-200 bg-white p-2 text-center">
                        <span className="font-mono text-base font-bold text-red-700 block tabular-nums">
                          {formatPct(overall?.error_rate)}
                        </span>
                        <span className="text-[8px] uppercase text-slate-500 tracking-wide">{t.errorRate}</span>
                      </div>
                      <div className="rounded border border-slate-200 bg-white p-2 text-center">
                        <span className="font-mono text-base font-bold text-teal-700 block tabular-nums">
                          {formatPct(highCov, 1)}
                        </span>
                        <span className="text-[8px] uppercase text-slate-500 tracking-wide">{t.highResidualCov}</span>
                      </div>
                      <div className="rounded border border-slate-200 bg-white p-2 text-center">
                        <span className="font-mono text-base font-bold text-slate-900 block tabular-nums">
                          {nPatterns ?? '—'}
                        </span>
                        <span className="text-[8px] uppercase text-slate-500 tracking-wide">{t.patternsLabel}</span>
                      </div>
                      <div className="rounded border border-slate-200 bg-white p-2 text-center sm:col-span-1 col-span-2">
                        <span className="font-mono text-base font-bold text-red-800 block tabular-nums">
                          {formatFixed(evRed?.avg_pairwise_jaccard)}
                        </span>
                        <span className="text-[8px] uppercase text-slate-500 tracking-wide">Avg Jaccard</span>
                      </div>
                    </div>
                  </PosterCard>

                  <p className="font-mono text-[9px] uppercase tracking-widest text-teal-800">{t.resultsTitle}</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {topSubjects.length > 0 ? (
                      <SubjectBarChart rows={topSubjects} title={t.figSubject} hint={t.subjectBarHint} />
                    ) : (
                      <PosterCard accent="none">
                        <p className="text-[10px] text-slate-500 italic">Subject breakdown loading…</p>
                      </PosterCard>
                    )}
                    <FigCard tag="Fig B" tagAccent title={t.figHeatmap} caption={t.figHeatmapCap}>
                      {heatmapJson.loading ? (
                        <p className="text-[10px] text-slate-500">{t.heatmapLoading}</p>
                      ) : heatmapJson.error || !heatmapJson.data?.subjects?.length ? (
                        <p className="text-[10px] text-slate-500">{t.heatmapError}</p>
                      ) : (
                        <SaeLatentDeltaHeatmap
                          data={heatmapJson.data}
                          legendLow={t.heatmapLegendLow}
                          legendHigh={t.heatmapLegendHigh}
                          nullLabel={t.heatmapNull}
                        />
                      )}
                    </FigCard>
                  </div>

                  <PosterCard accent="none" className="bg-teal-50/40 border-teal-100">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-teal-800 mb-2">{t.metricsExplain}</p>
                    <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center">
                      <div>
                        <div className="font-mono text-sm font-bold text-teal-800 tabular-nums">{formatPct(covSet, 1)}</div>
                        <div className="text-[8px] uppercase text-slate-600">{t.metricCov}</div>
                        <div className="text-[8px] text-slate-500 mt-0.5">{t.metricCovNote}</div>
                      </div>
                      <div>
                        <div className="font-mono text-sm font-bold text-amber-800 tabular-nums">{formatFixed(conc, 3)}</div>
                        <div className="text-[8px] uppercase text-slate-600">{t.metricConc}</div>
                        <div className="text-[8px] text-slate-500 mt-0.5">{t.metricConcNote}</div>
                      </div>
                      <div>
                        <div className="font-mono text-sm font-bold text-amber-800 tabular-nums">{formatFixed(evPU?.auc)}</div>
                        <div className="text-[8px] uppercase text-slate-600">{t.metricAuc}</div>
                        <div className="text-[8px] text-slate-500 mt-0.5">{t.metricAucNote}</div>
                      </div>
                      <div>
                        <div className="font-mono text-sm font-bold text-amber-800 tabular-nums">{formatFixed(evPU?.f1)}</div>
                        <div className="text-[8px] uppercase text-slate-600">{t.metricF1}</div>
                        <div className="text-[8px] text-slate-500 mt-0.5">{t.metricF1Note}</div>
                      </div>
                      <div>
                        <div className="font-mono text-sm font-bold text-slate-700 tabular-nums">{stabilityDisplay}</div>
                        <div className="text-[8px] uppercase text-slate-600">{t.metricStab}</div>
                        <div className="text-[8px] text-slate-500 mt-0.5">{t.metricStabNote}</div>
                      </div>
                    </div>
                    <ul className="list-disc pl-4 mt-2 text-[9px] text-slate-600 space-y-0.5">
                      <li>{interpAuc}</li>
                      <li>{interpJaccard}</li>
                    </ul>
                  </PosterCard>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
                    <FigCard tag="Fig F" tagAccent title={t.discPatternFig} caption={t.discPatternCaption}>
                      <PosterDiscoveredPattern
                        headline={discPatternHeadline}
                        body={t.discPatternBody}
                        tags={discPatternTags}
                        caption={t.patternDisclaimer}
                        empty={t.discPatternEmpty}
                        showPanel={!!topPattern}
                      />
                    </FigCard>
                    <FigCard tag="Fig G" title={t.setLevelFig} caption={t.setLevelCaption}>
                      <PosterSetLevelMetrics
                        coverage={covSet}
                        concentration={conc}
                        auc={evPU?.auc}
                        f1={evPU?.f1}
                        avgJaccard={evRed?.avg_pairwise_jaccard}
                        stabilityDetail={stabilityDetailLine}
                        labels={{
                          radarHelp: t.setLevelRadarHelp,
                          stabilityHeading: t.setLevelStabilityHeading,
                          stabilityNoData: t.setLevelStabilityNoData,
                          axisCov: t.setLevelAxisCov,
                          axisConc: t.setLevelAxisConc,
                          axisAuc: t.setLevelAxisAuc,
                          axisF1: t.setLevelAxisF1,
                          axisDistinct: t.setLevelAxisDistinct,
                          legendOur: t.setLevelLegendOur,
                          legendRef: t.setLevelLegendRef,
                          tableTitle: t.setLevelTable,
                        }}
                      />
                    </FigCard>
                  </div>

                </div>

                {/* RIGHT */}
                <div className="flex flex-col gap-3">
                  <PosterCard accent="amber" className="bg-gradient-to-br from-amber-50/80 to-white">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-amber-800 mb-1">{t.coreTitle}</p>
                    <p className="text-[10.5px] text-slate-700 border-l-2 border-amber-500 pl-2 italic leading-relaxed">
                      {t.coreQuote}
                    </p>
                  </PosterCard>

                  <PosterCard accent="teal">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-teal-800 mb-2">{t.storyTitle}</p>
                    <div className="space-y-2 text-[10px] text-slate-600">
                      <p>
                        <span className="text-teal-600 font-bold mr-1">①</span>
                        {nInst.toLocaleString()} items →{' '}
                        <strong className="text-amber-800">{formatPct(overall?.error_rate)}</strong> error rate on this MMLU pass.
                      </p>
                      <p>
                        <span className="text-teal-600 font-bold mr-1">②</span>
                        Difficulty model:{' '}
                        <strong className="text-amber-800">{formatPct(highCov, 1)}</strong> of errors lie in the high-residual cohort
                        (unexpected mistakes under shallow features).
                      </p>
                      <p>
                        <span className="text-teal-600 font-bold mr-1">③</span>
                        SAE diffing ranks {nPatterns ?? '—'} candidate patterns; top latent_id{' '}
                        <strong className="text-teal-800">{topPattern?.latent_id ?? '—'}</strong> is the strongest catalog hit this run.
                      </p>
                      <p>
                        <span className="text-teal-600 font-bold mr-1">④</span>
                        Subject bars show uneven error mass (e.g., STEM-heavy subjects often rank high)—structure beyond a single global rate.
                      </p>
                      <p>
                        <span className="text-teal-600 font-bold mr-1">⑤</span>
                        F1 {formatFixed(evPU?.f1)} vs AUC {formatFixed(evPU?.auc)}: some thresholded signal with weak ranking; Jaccard{' '}
                        {formatFixed(evRed?.avg_pairwise_jaccard)} flags redundancy.
                      </p>
                    </div>
                  </PosterCard>

                  <PosterCard>
                    <p className="font-mono text-[9px] uppercase tracking-widest text-slate-600 mb-2">{t.limitsTitle}</p>
                    <ul className="space-y-2 text-[10px] text-slate-600">
                      <li className="flex gap-2">
                        <span className="text-slate-400 shrink-0">•</span>
                        <span>{t.lim1}</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-slate-400 shrink-0">•</span>
                        <span>{t.lim2}</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-slate-400 shrink-0">•</span>
                        <span>{t.lim3}</span>
                      </li>
                    </ul>
                  </PosterCard>

                  <PosterCard accent="none">
                    <h2 className="font-headline text-[10px] font-black uppercase tracking-widest text-teal-800 mb-1.5">
                      {t.conclusions}
                    </h2>
                    <ul className="list-disc pl-4 space-y-1 text-[10px] text-slate-700">
                      <li>{t.c1}</li>
                      <li>{t.c2}</li>
                      <li>{t.c3}</li>
                      <li>{t.c4}</li>
                    </ul>
                  </PosterCard>

                  <PosterCard accent="none">
                    <h2 className="font-headline text-[10px] font-black uppercase tracking-widest text-teal-800 mb-1">
                      {t.references}
                    </h2>
                    <ol className="list-decimal pl-4 space-y-1 text-[8.5px] text-slate-600 leading-snug">
                      <li>{t.ref1}</li>
                      <li>{t.ref2}</li>
                      <li>{t.ref3}</li>
                    </ol>
                    <h2 className="font-headline text-[10px] font-black uppercase tracking-widest text-teal-800 mt-2 mb-1">
                      {t.acknowledgments}
                    </h2>
                    <p className="text-[9px] text-slate-600 leading-relaxed">{t.ackBody}</p>
                  </PosterCard>
                </div>
              </div>
            </div>
          </article>
        </LoadState>
      </div>
    </div>
  )
}
