/** Served by Vite middleware from repo `pipeline_2026/`. */
export const P = {
  releaseReadiness:
    '/pipeline_2026/stage2/s06_report/results/release_readiness.json',
  paperMmluSummary: '/pipeline_2026/stage1/results/paper_mmlu_summary.json',
  residualProfile:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_gpt35_residuals_profile.json',
  residualMeta:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_gpt35_residuals.meta.json',
  residualPng:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_gpt35_residuals_error_by_subject.png',
  exportMeta:
    '/pipeline_2026/stage2/s02_export/results/mmlu_hf_hidden.meta.json',
  saeMeta:
    '/pipeline_2026/stage2/s03_sae_encode/results/mmlu_hf_sae_latents.meta.json',
  diffDiagnostics:
    '/pipeline_2026/stage2/s04_diff/results/mmlu_hf_latents.diagnostics.json',
  patternCatalog:
    '/pipeline_2026/stage2/s04_diff/results/mmlu_hf_latents.pattern_catalog.json',
  evalMetrics:
    '/pipeline_2026/stage2/s05_eval/results/mmlu_gpt35_sae_latents_head_200_metrics.json',
  evalResidualBar:
    '/pipeline_2026/stage2/s05_eval/results/mmlu_gpt35_sae_latents_head_200_residual_bar.png',
  leaderboard: '/pipeline_2026/stage2/s05_eval/results/leaderboard.csv',
  reportHtml: '/pipeline_2026/stage2/s06_report/results/mmlu_report.html',
} as const

export function fileNameOnly(absOrRel: string): string {
  const parts = absOrRel.split(/[/\\]/)
  return parts[parts.length - 1] || absOrRel
}
