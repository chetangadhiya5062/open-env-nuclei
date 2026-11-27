#!/usr/bin/env bash
# CPU smoke test of every training script (1-2 steps each). Run from the repo root: bash training/smoke_test.sh
set -euo pipefail
PY=${PYTHON:-python}
$PY -m training.build_datasets --smoke
$PY -m training.verify_datasets --dir training/runs/smoke_datasets
$PY -m training.train_ppo --smoke --run-name smoke
$PY -m training.tune_ppo --smoke --run-name smoke
$PY -m training.evaluate_ppo --smoke --model training/runs/ppo_smoke/smoke/ppo_policy.npz
$PY -m training.train_sft --smoke --run-name smoke
$PY -m training.tune_sft --smoke --run-name smoke
$PY -m training.train_grpo --smoke --run-name smoke
$PY -m training.evaluate_llm --smoke --run-name smoke
$PY -m training.fe_benchmark --smoke --run-name smoke
echo "all training smoke tests passed"
