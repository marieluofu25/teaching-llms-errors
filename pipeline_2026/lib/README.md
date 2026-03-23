# `lib/` — shared Python

Imported by every stage:

- **`repo_paths.py`** — `pipeline_2026_root()`, `teaching_llms_errors_repo_root()` (for legacy `pipeline/` data).
- **`io_schema.py`** — column checks, row-count validation for residual CSVs and `.npz` alignment.

Stage-specific logic lives under `stageN/code/`.
