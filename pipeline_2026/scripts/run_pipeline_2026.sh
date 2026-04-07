#!/usr/bin/env bash
# Staged pipeline (2026): Stage 1 = paper baseline; Stage 2 = numbered segments (s01–s06).
# Run from anywhere:  bash pipeline_2026/scripts/run_pipeline_2026.sh [mode]
#
# Modes: mmlu | mathcamps | mmlu-demo | mmlu-real | mmlu-report | smoke
# Env: PYTHON, JUDGE_JSON, BUILD_HTML, BUILD_RELEASE_SUMMARY, TAB4_TOP_QUESTIONS,
#      EXPORT_*, SAE_*, RESIDUAL_CSV, ACTIVATIONS_NPZ, LATENTS_CSV, HTML_OUT, RELEASE_JSON

set -euo pipefail

P26="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P26"
export PYTHONPATH="${P26}${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

SPAPER="${P26}/stage1/results"
S01="${P26}/stage2/s01_residual/results"
S02="${P26}/stage2/s02_export/results"
S03="${P26}/stage2/s03_sae_encode/results"
S04="${P26}/stage2/s04_diff/results"
S05="${P26}/stage2/s05_eval/results"
S06="${P26}/stage2/s06_report/results"
mkdir -p "$SPAPER" "$S01" "$S02" "$S03" "$S04" "$S05" "$S06"

CFG="${P26}/config"
AUDIT_MOD="stage2.s06_report.code.generate_stage_audit_html"

# Args: results_dir, stage_tag (for CSV hide: s01_residual only), title, readme_path
run_audit() {
  local stage_dir="$1"
  local stage_tag="$2"
  local title="$3"
  local readme="$4"
  if [[ "${stage_tag}" == "s01_residual" ]]; then
    "${PY}" -m "${AUDIT_MOD}" \
      --stage "${stage_tag}" \
      --title "${title}" \
      --readme "${readme}" \
      --results-dir "${stage_dir}" \
      --csv-hide-columns "data_x,expl_embs,data_y,ai_preds,ai_scores,ai_expl,to_shows,metadata" \
      --max-cell-chars "240"
  else
    "${PY}" -m "${AUDIT_MOD}" \
      --stage "${stage_tag}" \
      --title "${title}" \
      --readme "${readme}" \
      --results-dir "${stage_dir}"
  fi
}

MODE="${1:-mmlu}"

run_eval() {
  local label="$1"
  local csv="$2"
  local membership="${3:-}"
  local catalog="${4:-}"
  local stability_catalogs="${5:-}"
  local eval_args=( -m stage2.s05_eval.code.evaluate_pattern_sets
    --residual-csv "${csv}"
    --label "${label}"
    --output-json "${S05}/${label}_metrics.json"
    --output-table "${S05}/leaderboard.csv"
  )
  if [[ -n "${membership}" && -f "${membership}" ]]; then
    eval_args+=( --pattern-membership "${membership}" )
  fi
  if [[ -n "${catalog}" && -f "${catalog}" ]]; then
    eval_args+=( --pattern-catalog "${catalog}" )
  fi
  if [[ -n "${stability_catalogs}" ]]; then
    eval_args+=( --stability-catalogs "${stability_catalogs}" )
  fi
  if [[ -n "${JUDGE_JSON:-}" && -f "${JUDGE_JSON}" ]]; then
    eval_args+=( --judge-json "${JUDGE_JSON}" )
  fi
  "${PY}" "${eval_args[@]}"
  "${PY}" -m stage2.s05_eval.code.plot_residual_summary \
    --metrics-json "${S05}/${label}_metrics.json" \
    --output "${S05}/${label}_residual_bar.png"
}

run_build_html() {
  local res="${RESIDUAL_CSV:-${S01}/mmlu_gpt35_residuals.csv}"
  local out="${HTML_OUT:-${S06}/mmlu_report.html}"
  local npz="${ACTIVATIONS_NPZ:-}"
  local lat="${LATENTS_CSV:-}"
  if [[ ! -f "${res}" ]]; then
    echo "run_build_html: missing residual CSV: ${res}" >&2
    return 1
  fi
  echo "==> Five-tab HTML report → ${out} …"
  local html_args=( -m stage2.s06_report.code.build_html_report
    --residual-csv "${res}"
    --meanings-yaml "${CFG}/feature_meanings.yaml"
    --output "${out}"
  )
  if [[ -n "${npz}" && -f "${npz}" ]]; then
    html_args+=( --activations "${npz}" )
    if [[ -n "${lat}" && -f "${lat}" ]]; then
      html_args+=( --latents-csv "${lat}" )
    fi
  fi
  if [[ -n "${TAB4_TOP_QUESTIONS:-}" ]]; then
    html_args+=( --tab4-top-questions "${TAB4_TOP_QUESTIONS}" )
  fi
  if [[ -f "${RELEASE_JSON:-${S06}/release_readiness.json}" ]]; then
    html_args+=( --release-json "${RELEASE_JSON:-${S06}/release_readiness.json}" )
  fi
  "${PY}" "${html_args[@]}"
}

run_release_summary() {
  local res="${1:?need residual CSV}"
  local out="${RELEASE_JSON:-${S06}/release_readiness.json}"
  echo "==> Release summary → ${out} …"
  local rs_args=( -m stage2.s06_report.code.release_summary --residual-csv "${res}" --output "${out}" )
  if [[ -n "${METRICS_JSON:-}" && -f "${METRICS_JSON}" ]]; then
    rs_args+=( --metrics-json "${METRICS_JSON}" )
  fi
  if [[ -n "${ACTIVATIONS_NPZ:-}" && -f "${ACTIVATIONS_NPZ}" ]]; then
    meta="${ACTIVATIONS_NPZ%.npz}.meta.json"
    if [[ -f "${meta}" ]]; then
      rs_args+=( --activations-meta "${meta}" )
    fi
  fi
  "${PY}" "${rs_args[@]}"
}

run_paper_baseline() {
  echo "==> Stage 1 — Paper baseline (figures + summary)…"
  "${PY}" -m stage1.code.run_paper_baseline \
    --output-dir "${SPAPER}" || echo "    (paper baseline skipped or failed — see stderr)" >&2
  run_audit "${SPAPER}" "paper_baseline" "Stage 1 — Paper baseline" "${P26}/stage1/README.md"
}

case "${MODE}" in
  smoke)
    exec bash "${P26}/scripts/smoke_test.sh"
    ;;
  mmlu)
    run_paper_baseline
    echo "==> Stage 2 — s01_residual — Residual control (MMLU, train_only)…"
    "${PY}" -m stage2.s01_residual.code.run_residual_control \
      --dataset mmlu \
      --output "${S01}/mmlu_gpt35_residuals.csv" \
      --fit-mode train_only
    run_audit "${S01}" "s01_residual" "Stage 2 — s01_residual — Residual control" "${P26}/stage2/s01_residual/README.md"
    echo "==> Stage 2 — s05_eval — Evaluate + plot…"
    run_eval "mmlu_gpt35_train_only" "${S01}/mmlu_gpt35_residuals.csv"
    run_audit "${S05}" "s05_eval" "Stage 2 — s05_eval — Set-level metrics" "${P26}/stage2/s05_eval/README.md"
    if [[ "${BUILD_RELEASE_SUMMARY:-}" == "1" ]]; then
      METRICS_JSON="${S05}/mmlu_gpt35_train_only_metrics.json" \
        run_release_summary "${S01}/mmlu_gpt35_residuals.csv"
      run_audit "${S06}" "s06_report" "Stage 2 — s06_report — Reporting" "${P26}/stage2/s06_report/README.md"
    fi
    if [[ "${BUILD_HTML:-}" == "1" ]]; then
      RESIDUAL_CSV="${S01}/mmlu_gpt35_residuals.csv" \
      HTML_OUT="${S06}/mmlu_report.html" \
      RELEASE_JSON="${S06}/release_readiness.json" \
        run_build_html || true
      run_audit "${S06}" "s06_report" "Stage 2 — s06_report — Reporting" "${P26}/stage2/s06_report/README.md"
    fi
    echo "Done. See stage1/results/, stage2/s01_residual/results/, s05_eval/, s06_report/ + audit.html."
    ;;
  mathcamps)
    run_paper_baseline
    echo "==> Stage 2 — s01_residual — Residual control (MathCAMPs)…"
    "${PY}" -m stage2.s01_residual.code.run_residual_control \
      --dataset mathcamps \
      --model-filter "gpt-3.5-turbo-0125" \
      --output "${S01}/mathcamps_gpt35_residuals.csv" \
      --fit-mode train_only
    run_audit "${S01}" "s01_residual" "Stage 2 — s01_residual — Residual control" "${P26}/stage2/s01_residual/README.md"
    echo "==> Stage 2 — s05_eval — Evaluate + plot…"
    run_eval "mathcamps_gpt35_train_only" "${S01}/mathcamps_gpt35_residuals.csv"
    run_audit "${S05}" "s05_eval" "Stage 2 — s05_eval — Set-level metrics" "${P26}/stage2/s05_eval/README.md"
    echo "Done."
    ;;
  mmlu-demo)
    run_paper_baseline
    echo "==> Stage 2 — s01_residual — MMLU residual…"
    "${PY}" -m stage2.s01_residual.code.run_residual_control \
      --dataset mmlu \
      --output "${S01}/mmlu_gpt35_residuals.csv" \
      --fit-mode train_only
    run_audit "${S01}" "s01_residual" "Stage 2 — s01_residual — Residual control" "${P26}/stage2/s01_residual/README.md"
    echo "==> Stage 2 — s04_diff — Dummy activations + Welch diff…"
    "${PY}" -m stage2.s04_diff.code.generate_dummy_activations \
      --residual-csv "${S01}/mmlu_gpt35_residuals.csv" \
      --latent-dim 64 \
      --output "${S04}/mmlu_dummy_activations.npz"
    "${PY}" -m stage2.s04_diff.code.run_sae_diffing \
      --activations "${S04}/mmlu_dummy_activations.npz" \
      --residual-csv "${S01}/mmlu_gpt35_residuals.csv" \
      --min-group-size 30 \
      --output "${S04}/mmlu_dummy_latents.csv" || true
    run_audit "${S04}" "s04_diff" "Stage 2 — s04_diff — Coordinate-wise group comparison" "${P26}/stage2/s04_diff/README.md"
    echo "==> Stage 2 — s05_eval — Evaluate + plot…"
    run_eval "mmlu_gpt35_demo" "${S01}/mmlu_gpt35_residuals.csv"
    run_audit "${S05}" "s05_eval" "Stage 2 — s05_eval — Set-level metrics" "${P26}/stage2/s05_eval/README.md"
    if [[ "${BUILD_HTML:-}" == "1" ]]; then
      RESIDUAL_CSV="${S01}/mmlu_gpt35_residuals.csv"
      ACTIVATIONS_NPZ="${ACTIVATIONS_NPZ:-${S04}/mmlu_dummy_activations.npz}"
      LATENTS_CSV="${LATENTS_CSV:-${S04}/mmlu_dummy_latents.csv}"
      HTML_OUT="${S06}/mmlu_report.html"
      run_build_html || true
      run_audit "${S06}" "s06_report" "Stage 2 — s06_report — Reporting" "${P26}/stage2/s06_report/README.md"
    fi
    echo "Done (demo)."
    ;;
  mmlu-report)
    RESIDUAL_CSV="${RESIDUAL_CSV:-${S01}/mmlu_gpt35_residuals.csv}"
    HTML_OUT="${HTML_OUT:-${S06}/mmlu_report.html}"
    run_build_html
    run_audit "${S06}" "s06_report" "Stage 2 — s06_report — Reporting" "${P26}/stage2/s06_report/README.md"
    echo "Done (HTML report)."
    ;;
  mmlu-real)
    run_paper_baseline
    echo "==> Stage 2 — s01_residual — Residual control…"
    "${PY}" -m stage2.s01_residual.code.run_residual_control \
      --dataset mmlu \
      --output "${S01}/mmlu_gpt35_residuals.csv" \
      --fit-mode train_only
    run_audit "${S01}" "s01_residual" "Stage 2 — s01_residual — Residual control" "${P26}/stage2/s01_residual/README.md"
    RES_CSV="${S01}/mmlu_gpt35_residuals.csv"
    RES_WORK="${RES_CSV}"
    if [[ -n "${EXPORT_MAX_ROWS:-}" ]]; then
      RES_WORK="${S01}/mmlu_gpt35_residuals_export_${EXPORT_MAX_ROWS}.csv"
      echo "==> Subset residual CSV → ${RES_WORK}"
      "${PY}" -c "import pandas as pd; df=pd.read_csv('${RES_CSV}'); df.iloc[:int(${EXPORT_MAX_ROWS})].to_csv('${RES_WORK}', index=False)"
      run_audit "${S01}" "s01_residual" "Stage 2 — s01_residual — Residual control" "${P26}/stage2/s01_residual/README.md"
    fi
    HIDDEN_NPZ="${S02}/mmlu_hf_hidden.npz"
    SAE_NPZ="${S03}/mmlu_hf_sae_latents.npz"
    LAT_OUT="${S04}/mmlu_hf_latents.csv"
    SAE_ENCODE="${SAE_ENCODE:-1}"
    if [[ "${SAE_ENCODE}" == "1" && -z "${EXPORT_LAYER_INDEX:-}" && "${EXPORT_MODEL:-gpt2}" == "gpt2" ]]; then
      EXPORT_LAYER_INDEX=11
      echo "    (default EXPORT_LAYER_INDEX=11 for blocks.11.hook_resid_pre)"
    elif [[ "${SAE_ENCODE}" == "1" && -z "${EXPORT_LAYER_INDEX:-}" && "${EXPORT_MODEL:-gpt2}" != "gpt2" ]]; then
      echo "WARNING: set EXPORT_LAYER_INDEX to match SAE_ID (docs/sae_checkpoints.md)" >&2
    fi
    echo "==> Stage 2 — s02_export — Export HF hidden states → ${HIDDEN_NPZ}"
    EXP=( -m stage2.s02_export.code.export_activations --residual-csv "${RES_WORK}" --output "${HIDDEN_NPZ}" )
    if [[ -n "${EXPORT_MODEL:-}" ]]; then EXP+=( --model "${EXPORT_MODEL}" ); fi
    if [[ -n "${EXPORT_LAYER_INDEX:-}" ]]; then EXP+=( --layer-index "${EXPORT_LAYER_INDEX}" ); fi
    if [[ -n "${EXPORT_BATCH_SIZE:-}" ]]; then EXP+=( --batch-size "${EXPORT_BATCH_SIZE}" ); fi
    if [[ -n "${EXPORT_DEVICE:-}" ]]; then EXP+=( --device "${EXPORT_DEVICE}" ); fi
    if [[ "${EXPORT_FP16:-}" == "1" ]]; then EXP+=( --fp16 ); fi
    "${PY}" "${EXP[@]}"
    run_audit "${S02}" "s02_export" "Stage 2 — s02_export — Surrogate activations (HF)" "${P26}/stage2/s02_export/README.md"
    DIFF_NPZ="${HIDDEN_NPZ}"
    if [[ "${SAE_ENCODE}" == "1" ]]; then
      echo "==> Stage 2 — s03_sae_encode — SAELens encode → ${SAE_NPZ}"
      ENC=( -m stage2.s03_sae_encode.code.encode_sae_latents --input-npz "${HIDDEN_NPZ}" --output-npz "${SAE_NPZ}" )
      if [[ -n "${SAE_RELEASE:-}" ]]; then ENC+=( --sae-release "${SAE_RELEASE}" ); fi
      if [[ -n "${SAE_ID:-}" ]]; then ENC+=( --sae-id "${SAE_ID}" ); fi
      if [[ -n "${SAE_BATCH_SIZE:-}" ]]; then ENC+=( --batch-size "${SAE_BATCH_SIZE}" ); fi
      if [[ -n "${EXPORT_DEVICE:-}" ]]; then ENC+=( --device "${EXPORT_DEVICE}" ); fi
      "${PY}" "${ENC[@]}"
      DIFF_NPZ="${SAE_NPZ}"
      run_audit "${S03}" "s03_sae_encode" "Stage 2 — s03_sae_encode — SAE latent vectors" "${P26}/stage2/s03_sae_encode/README.md"
    else
      echo "==> SAE_ENCODE=0: skip s03_sae_encode"
    fi
    echo "==> Stage 2 — s04_diff — Welch diff on coordinates…"
    MEMBERSHIP_MAIN="${S04}/mmlu_hf_latents.pattern_membership.csv"
    CATALOG_MAIN="${S04}/mmlu_hf_latents.pattern_catalog.json"
    "${PY}" -m stage2.s04_diff.code.run_sae_diffing \
      --activations "${DIFF_NPZ}" \
      --residual-csv "${RES_WORK}" \
      --min-group-size 30 \
      --output-membership "${MEMBERSHIP_MAIN}" \
      --output-pattern-catalog "${CATALOG_MAIN}" \
      --output "${LAT_OUT}" || true
    run_audit "${S04}" "s04_diff" "Stage 2 — s04_diff — Coordinate-wise group comparison" "${P26}/stage2/s04_diff/README.md"
    EVAL_TAG="mmlu_gpt35_hf_activations"
    if [[ "${SAE_ENCODE}" == "1" ]]; then
      EVAL_TAG="mmlu_gpt35_sae_latents"
    fi
    if [[ -n "${EXPORT_MAX_ROWS:-}" ]]; then
      EVAL_TAG="${EVAL_TAG}_head_${EXPORT_MAX_ROWS}"
    fi
    STABILITY_CATALOGS=""
    if [[ -n "${STABILITY_SEEDS:-}" ]]; then
      echo "==> Stage 2 — stability runs over seeds: ${STABILITY_SEEDS}"
      IFS=',' read -r -a SEEDS <<< "${STABILITY_SEEDS}"
      for seed in "${SEEDS[@]}"; do
        seed="$(echo "${seed}" | xargs)"
        [[ -z "${seed}" ]] && continue
        RES_SEED="${S01}/mmlu_gpt35_residuals_seed_${seed}.csv"
        LAT_SEED="${S04}/mmlu_hf_latents_seed_${seed}.csv"
        MEM_SEED="${S04}/mmlu_hf_latents_seed_${seed}.pattern_membership.csv"
        CAT_SEED="${S04}/mmlu_hf_latents_seed_${seed}.pattern_catalog.json"
        "${PY}" -m stage2.s01_residual.code.run_residual_control \
          --dataset mmlu \
          --output "${RES_SEED}" \
          --fit-mode train_only \
          --random-state "${seed}"
        "${PY}" -m stage2.s04_diff.code.run_sae_diffing \
          --activations "${DIFF_NPZ}" \
          --residual-csv "${RES_SEED}" \
          --min-group-size 30 \
          --output-membership "${MEM_SEED}" \
          --output-pattern-catalog "${CAT_SEED}" \
          --output "${LAT_SEED}" || true
        run_eval "${EVAL_TAG}_seed_${seed}" "${RES_SEED}" "${MEM_SEED}" "${CAT_SEED}"
        STABILITY_CATALOGS="${STABILITY_CATALOGS}${STABILITY_CATALOGS:+,}${CAT_SEED}"
      done
    fi
    echo "==> Stage 2 — s05_eval — Evaluate + plot…"
    run_eval "${EVAL_TAG}" "${RES_WORK}" "${MEMBERSHIP_MAIN}" "${CATALOG_MAIN}" "${STABILITY_CATALOGS}"
    run_audit "${S05}" "s05_eval" "Stage 2 — s05_eval — Set-level metrics" "${P26}/stage2/s05_eval/README.md"
    export METRICS_JSON="${S05}/${EVAL_TAG}_metrics.json"
    export ACTIVATIONS_NPZ="${DIFF_NPZ}"
    RELEASE_JSON="${S06}/release_readiness.json"
    run_release_summary "${RES_WORK}"
    RESIDUAL_CSV="${RES_WORK}"
    LATENTS_CSV="${LAT_OUT}"
    HTML_OUT="${S06}/mmlu_report.html"
    run_build_html
    run_audit "${S06}" "s06_report" "Stage 2 — s06_report — Reporting" "${P26}/stage2/s06_report/README.md"
    echo "Done (mmlu-real). Open stage1/results/ and stage2/s*/results/audit.html; full report: ${S06}/mmlu_report.html"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Use: $0 [mmlu|mathcamps|mmlu-demo|mmlu-real|mmlu-report|smoke]" >&2
    exit 1
    ;;
esac
