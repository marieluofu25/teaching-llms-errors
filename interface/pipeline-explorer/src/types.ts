export type ReleaseReadiness = {
  behavioral_model?: { mode?: string; unique_values?: string[] }
  representation_model?: {
    model?: string
    n_rows?: number
    d_in?: number
    d_sae?: number
    sae_release?: string
    sae_id?: string
    output_npz?: string
    source_hidden_meta?: string
    residual_csv?: string
    pattern_catalog_source?: string
  }
  overall?: { n_instances?: number; n_errors?: number; error_rate?: number; accuracy?: number }
  gates?: { verdict?: string; violations?: { gate?: string; detail?: string }[] }
  final_set_level_metrics?: {
    predictive_utility?: { auc?: number; f1?: number; n_features?: number }
    coverage_concentration?: { coverage?: number; concentration?: number }
    redundancy?: { avg_pairwise_jaccard?: number; n_patterns?: number }
  }
}

export type ResidualProfile = {
  dataset?: string
  n_rows?: number
  error_rate?: number
  n_errors?: number
  group_counts?: Record<string, number>
  expected_error_mean?: number
  residual_error_quantiles?: { q25?: number; q50?: number; q75?: number }
  by_subject?: { subject: string; n: number; errors: number; error_rate: number }[]
}

export type ExportMeta = {
  model?: string
  n_rows?: number
  hidden_dim?: number
  layer_index?: number
  output_npz?: string
}

export type SaeMeta = {
  model?: string
  n_rows?: number
  d_sae?: number
  sae_release?: string
  sae_id?: string
  output_npz?: string
}

export type DiffDiagnostics = {
  n_high_residual?: number
  n_low_residual?: number
  n_latents_tested?: number
  n_passing?: number
}

export type PatternCatalog = {
  n_rows?: number
  n_patterns?: number
  membership_quantile?: number
  patterns?: {
    pattern_id: number
    latent_id: number
    threshold: number
    V_prime: number
    p_value: number
  }[]
}

export type EvalMetrics = {
  label?: string
  residual_metrics?: {
    n_instances?: number
    n_errors?: number
    error_rate?: number
    high_residual?: { n?: number; coverage_of_errors?: number }
  }
  set_level_metrics?: {
    predictive_utility?: { auc?: number; f1?: number }
    coverage_concentration?: { coverage?: number; concentration?: number }
    n_patterns?: number
  }
}

export type PaperMmluSummary = {
  n_rows?: number
  error_rate?: number
  figures_written?: string[]
}
