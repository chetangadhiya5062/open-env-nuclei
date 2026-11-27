# Changelog

## 0.4.1 (branch v4.1-cleanup)

### Changed
- Product packages moved to the optional `[app]` extra (`pip install -e ".[app]"`); the core install and the
  Hugging Face Space `requirements.txt` contain only the environment's dependencies. `[dev]` includes `[app]`.
- Leak handling is a setting: `leak_policy: auto | drop | warn` (recipe step `drop_columns`, `auto_recipe`,
  `fe auto/run --leak-policy`, API `leak_policy`, dashboard toggle). Default `auto` drops high-confidence leaks and
  warns about medium-confidence ones. Findings and actions are listed in the step report, run summary, HTML report
  ("Target leak check"), dataset card and CLI output.
- 4-bit (QLoRA) model loading for GRPO and LLM evaluation as well as SFT; runbook section "Choosing the model" with
  VRAM guidance and the Llama licence steps. Default model unchanged (Qwen2.5-1.5B-Instruct).
- Removed references to the coding tool from older docs.

### Added
- `featureforge.profile.detect_leaks` (confidence levels), 9 tests (192 total).

## 0.4.0 (branch v4-feature-engineering, unreleased)

### Added
- `featureforge` library: file reading (CSV/TSV/Excel/Parquet/JSON/JSONL/Feather), semantic type inference,
  profiling with quality score, leak checks and drift (PSI + KS), 20 recipe steps (cleaning, feature engineering,
  encoding, selection, OpenFE-style generation), fit/transform pipeline with input validation and per-step reports,
  saved artifacts (recipe.yaml, pipeline.joblib, metadata.json), auto recipes (clean / standard / full),
  leakage-safe train/test split, HTML quality report, dataset card, synthetic messy sample datasets.
- Gradio dashboard (`/ui`), FastAPI REST API (`/v1`), CLI `fe`.
- Product Docker image (`docker/featureforge.Dockerfile`) and `docker-compose.yml`.
- `featureforge.envbridge` and `scripts/eval_library_on_env.py`: the library's cleaning graded by the environment's
  grader (0.967 / 0.960 / 0.932 on test seeds 0-9).
- `training/`: deterministic SFT / GRPO / tabular datasets with manifest and checksums; PPO, LoRA SFT, GRPO, Optuna
  and evaluation scripts with YAML configs; feature-engineering validation benchmark; `GPU_RUNBOOK.md`; CPU smoke
  tests. No training was run.
- `docs/RESEARCH.md` and `docs/references/` (index, citation notes, fetch script).
- 86 tests (183 total). CI: CLI smoke run, training smoke job, product image build and API check.

### Changed
- README rewritten as a product page; the environment documentation is kept.
- Package version 0.4.0; new runtime dependencies (scikit-learn, scipy, pyarrow, openpyxl, joblib, pyyaml, jinja2,
  matplotlib, typer, python-multipart).

## 0.3.0 (v3)
- PPO agent (400k steps) with NumPy inference, LLM benchmark (Llama 3.1 8B), improvement metric, harder easy task,
  Hugging Face Space deploy script.

## 0.2.0 (v2)
- Rewrite: seeded tasks with hidden ground truth, typed actions, potential-based reward, grader, agents, benchmark,
  tests, CI, Gradio demo.
