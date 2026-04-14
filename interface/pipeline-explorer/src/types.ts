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
    base_model?: string
    methodology?: string
  }
  overall?: { n_instances?: number; n_errors?: number; error_rate?: number; accuracy?: number }
  per_subject?: { subject: string; n: number; error_rate: number; accuracy?: number }[]
  residual_metrics_from_eval?: {
    n_instances?: number
    n_errors?: number
    error_rate?: number
    high_residual?: { n?: number; coverage_of_errors?: number; concentration_in_group?: number }
    low_residual?: { n?: number; coverage_of_errors?: number; concentration_in_group?: number }
  }
  gates?: { verdict?: string; violations?: { gate?: string; detail?: string }[] }
  final_set_level_metrics?: {
    predictive_utility?: { auc?: number; f1?: number; n_features?: number; split_seed?: number }
    coverage_concentration?: {
      coverage?: number
      concentration?: number
      n_errors?: number
      n_covered_errors?: number
      n_covered_instances?: number
    }
    redundancy?: { avg_pairwise_jaccard?: number; n_patterns?: number; max_pairwise_jaccard?: number }
    stability?: { top_k?: number; pairs_evaluated?: number; overlap_at_k_mean?: number | null }
    n_patterns?: number
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

/** Binned residual_error for poster histogram (see build_residual_histogram_json.py). */
export type ResidualHistogram = {
  column?: string
  n_rows?: number
  n_bins?: number
  min?: number
  max?: number
  edges?: number[]
  counts?: number[]
  quantiles?: { q25?: number; q50?: number; q75?: number }
  group_split_quantiles?: { high_q?: number; low_q?: number }
  full_range?: { min?: number; max?: number; n_bins?: number }
  full?: { min?: number; max?: number; n_bins?: number; edges?: number[]; counts?: number[] }
  zoom?: {
    percentile_window?: [number, number]
    min?: number
    max?: number
    n_bins?: number
    edges?: number[]
    counts?: number[]
  }
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

/** Subject × top-K latent mean(High) − mean(Low) SAE activation (see build_subject_latent_heatmap.py). */
export type SubjectLatentDeltaHeatmap = {
  unit?: string
  min_per_side?: number
  n_rows_total?: number
  d_sae?: number
  subjects: string[]
  latent_ids: number[]
  values: (number | null)[][]
  n_high?: number[][]
  n_low?: number[][]
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
    predictive_utility?: { auc?: number; f1?: number; n_features?: number }
    coverage_concentration?: { coverage?: number; concentration?: number }
    n_patterns?: number
    redundancy?: {
      avg_pairwise_jaccard?: number
      n_patterns?: number
      max_pairwise_jaccard?: number
    }
  }
}

export type PaperMmluSummary = {
  n_rows?: number
  error_rate?: number
  figures_written?: string[]
}
