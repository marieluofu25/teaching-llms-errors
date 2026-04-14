/** Served by Vite middleware from repo `pipeline_2026/`. */
export const P = {
  releaseReadiness:
    '/pipeline_2026/stage2/s06_report/results/release_readiness.json',
  paperMmluSummary: '/pipeline_2026/stage1/results/paper_mmlu_summary.json',
  residualProfile:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_full_residuals_profile.json',
  mmluFullResidualProfile:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_full_residuals_profile.json',
  residualMeta:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_full_residuals.meta.json',
  residualHistogram:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_full_residuals.histogram.json',
  residualPng:
    '/pipeline_2026/stage2/s01_residual/results/mmlu_full_residuals_error_by_subject.png',
  exportMeta:
    '/pipeline_2026/stage2/s02_export/results/mmlu_full_hidden.meta.json',
  exportHiddenViz:
    '/pipeline_2026/stage2/s02_export/results/mmlu_full_hidden_viz.png',
  saeMeta:
    '/pipeline_2026/stage2/s03_sae_encode/results/mmlu_full_sae_latents.meta.json',
  saeTopLatentsViz:
    '/pipeline_2026/stage2/s03_sae_encode/results/mmlu_full_sae_top_latents.png',
  saeResearchViz:
    '/pipeline_2026/stage2/s03_sae_encode/results/mmlu_full_sae_latents_research_viz.png',
  saeGroupCompareViz:
    '/pipeline_2026/stage2/s03_sae_encode/results/mmlu_full_sae_group_compare.png',
  saeGroupScatterViz:
    '/pipeline_2026/stage2/s03_sae_encode/results/mmlu_full_sae_group_scatter.png',
  diffDiagnostics:
    '/pipeline_2026/stage2/s04_diff/results/mmlu_full_latents.diagnostics.json',
  patternCatalog:
    '/pipeline_2026/stage2/s04_diff/results/mmlu_full_latents.pattern_catalog.json',
  subjectLatentDeltaHeatmap:
    '/pipeline_2026/stage2/s04_diff/results/mmlu_full_latents.subject_latent_delta_heatmap.json',
  diffSignificanceHist:
    '/pipeline_2026/stage2/s04_diff/results/mmlu_full_diff_significance_hist.png',
  diffEffectMap:
    '/pipeline_2026/stage2/s04_diff/results/mmlu_full_diff_effect_map.png',
  evalMetrics:
    '/pipeline_2026/stage2/s05_eval/results/mmlu_final_metrics.json',
  evalResidualBar:
    '/pipeline_2026/stage2/s05_eval/results/mmlu_gemma_unified_sae_residual_bar.png',
  leaderboard: '/pipeline_2026/stage2/s05_eval/results/leaderboard.csv',
  reportHtml: '/pipeline_2026/stage2/s06_report/results/mmlu_report.html',
} as const

export function fileNameOnly(absOrRel: string): string {
  const parts = absOrRel.split(/[/\\]/)
  return parts[parts.length - 1] || absOrRel
}
