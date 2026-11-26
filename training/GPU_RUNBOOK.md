# GPU runbook

Everything here is prepared and smoke-tested on CPU (1-2 steps per script, `training/smoke_test.ps1`). Nothing in this
folder has been trained, tuned or validated for real yet: there are **no result numbers** until you run the jobs
below. Run every command from the repository root.

| Job | Script | Config | Uses | Output |
|---|---|---|---|---|
| A. PPO policy for the environment | `training.train_ppo` | `configs/ppo.yaml` | CPU cores (GPU optional) | `runs/ppo/<run>/ppo_policy.npz` |
| A'. PPO hyper-parameter search | `training.tune_ppo` | `configs/ppo_optuna.yaml` | CPU cores | `runs/ppo_optuna/<run>/best.json` |
| B. LLM supervised fine-tuning (LoRA) | `training.train_sft` | `configs/sft.yaml` | GPU | `runs/sft/<run>/adapter` |
| B'. SFT hyper-parameter search | `training.tune_sft` | `configs/sft_optuna.yaml` | GPU | `runs/sft_optuna/<run>/best.json` |
| C. GRPO with the environment as reward | `training.train_grpo` | `configs/grpo.yaml` | GPU | `runs/grpo/<run>/adapter` |
| D. LLM agent evaluation (test seeds 0-9) | `training.evaluate_llm` | `configs/eval_llm.yaml` | GPU | `runs/eval_llm/<run>/results.md` |
| E. PPO evaluation (test seeds 0-9) | `training.evaluate_ppo` | - | CPU | `<ppo run>/test_results.md` |
| F. Feature-engineering validation benchmark | `training.fe_benchmark` | `configs/fe_benchmark.yaml` | CPU cores | `runs/fe_benchmark/<run>/results.md` |

Seed protocol (never change it between runs): environment test seeds **0-9** are only used by D and E; PPO validation
seeds 50-59; SFT validation 60-89; SFT train 100-699; GRPO 1000-1499; PPO training samples seeds 100-99,999 at random.
Tabular benchmark: `*_train.parquet` for CV and tuning, `*_holdout.parquet` scored once at the end.

## 0. Check the machine

```bash
nvidia-smi                      # driver + CUDA version, free memory
python3 --version               # 3.10, 3.11 or 3.12
df -h .                         # ~30 GB free for model downloads and checkpoints
nproc                           # CPU cores (PPO and the FE benchmark use them)
```

Memory guide for job B/C with the default `Qwen/Qwen2.5-1.5B-Instruct` + LoRA in bf16: about 12-20 GB GPU memory with
`per_device_train_batch_size: 4` and gradient checkpointing (estimate; check with `nvidia-smi` during the first
minutes). For 7-8B models use `model.load_in_4bit=true` (QLoRA, needs `bitsandbytes`).

## 1. Get the code and install

```bash
git clone https://github.com/chetangadhiya5062/open-env-nuclei.git
cd open-env-nuclei
git checkout v4-feature-engineering      # once the branch is pushed; otherwise copy the project folder over

python3 -m venv .venv
source .venv/bin/activate                # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the CUDA version from nvidia-smi
pip install -e ".[dev]"
pip install -r training/requirements-gpu.txt
# optional (Linux): pip install bitsandbytes   # QLoRA

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m training.verify_datasets                 # 11/11 files OK (checksums match the manifest)
bash training/smoke_test.sh                        # ~2-4 min on CPU; must end with "all training smoke tests passed"
pytest -q                                          # optional: full test suite
```

Hugging Face models are downloaded on first use. Qwen2.5 needs no login. For `meta-llama/Llama-3.1-8B-Instruct`
accept the license on its model page, then run `hf auth login` (paste the token into that prompt, never into a file
that is committed).

Use `tmux` (or `nohup ... &`) so jobs survive a disconnect, and keep the logs:

```bash
tmux new -s train
mkdir -p training/logs
```

## 2. Job A: PPO (environment policy)

The policy is a 64x64 MLP, so it is CPU-bound; give it many environment workers. Start with the search (optional),
then the final run.

```bash
# A'. search: 40 trials x 300k steps, pruned on validation seeds; resumable (same --run-name)
python -m training.tune_ppo --run-name search1 --set base_overrides.n_envs=16 2>&1 | tee training/logs/tune_ppo.log
cat training/runs/ppo_optuna/search1/best.json        # contains "set_args"

# A. final run (use the set_args printed by the search, or the defaults)
python -m training.train_ppo --run-name final --set n_envs=16 total_timesteps=2000000 2>&1 | tee training/logs/ppo.log

# E. test-seed evaluation next to do-nothing and rule-based
python -m training.evaluate_ppo --model training/runs/ppo/final/ppo_policy.npz
```

To ship a better policy: copy `training/runs/ppo/final/ppo_policy.npz` and `.json` over `models/`, run
`python benchmark.py` and update the README table with the new numbers.

## 3. Jobs B-D: LLM agent

```bash
# D0. zero-shot baseline of the base model, so the fine-tune has something to beat
python -m training.evaluate_llm --run-name base-zeroshot 2>&1 | tee training/logs/eval_base.log

# B'. optional SFT search (16 short trials, objective = validation loss)
python -m training.tune_sft --run-name search1 2>&1 | tee training/logs/tune_sft.log
cat training/runs/sft_optuna/search1/best.json         # "set_args"

# B. SFT (2 epochs over ~14.4k examples)
python -m training.train_sft --run-name sft1 2>&1 | tee training/logs/sft.log
#    with the search result:  python -m training.train_sft --run-name sft1 --set train.learning_rate=... lora.r=...
#    bigger model:            --set model.name=Qwen/Qwen2.5-7B-Instruct model.load_in_4bit=true

# D1. evaluate the SFT adapter on test seeds 0-9
python -m training.evaluate_llm --run-name sft1 --set adapter=training/runs/sft/sft1/adapter

# C. GRPO starting from the SFT adapter (same base model!)
python -m training.train_grpo --run-name grpo1 --set model.adapter=training/runs/sft/sft1/adapter 2>&1 | tee training/logs/grpo.log

# D2. evaluate the GRPO adapter
python -m training.evaluate_llm --run-name grpo1 --set adapter=training/runs/grpo/grpo1/adapter
```

Several GPUs: prefix the training command with `accelerate launch --num_processes <n>` (run `accelerate config`
once). Monitor: `--set report_to=tensorboard`, then `tensorboard --logdir training/runs`.

## 4. Job F: feature-engineering validation benchmark (CPU)

Can run on the CPU cores at the same time as the GPU jobs.

```bash
python -m training.fe_benchmark --run-name v1 2>&1 | tee training/logs/fe_benchmark.log
# with Optuna tuning of the gradient-boosting model per dataset x pipeline variant (much slower):
python -m training.fe_benchmark --run-name v1-tuned --set tune.enabled=true
```

## 5. Bring the results back

Copy back (small files only; checkpoints stay on the GPU machine):

```
training/runs/ppo/final/{ppo_policy.npz,ppo_policy.json,learning_curve.png,test_results.md}
training/runs/ppo_optuna/search1/{best.json,trials.csv}
training/runs/sft_optuna/search1/{best.json,trials.csv}
training/runs/sft/sft1/{run.json,adapter/}            # adapter is ~10-80 MB
training/runs/grpo/grpo1/{run.json,adapter/}
training/runs/eval_llm/*/{results.md,episodes.csv,run.json}
training/runs/fe_benchmark/v1/{results.md,results.csv,run.json}
training/logs/
```

Then put the measured numbers (and how they were run) into the README and `docs/PHASE_NOTES.md`. Every `run.json`
records the config, library versions and GPU name.

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | halve `train.per_device_train_batch_size` and double `gradient_accumulation_steps`; `model.load_in_4bit=true`; lower `train.max_length` (prompts are ~1k tokens) |
| GRPO: `batch size must be divisible by num_generations` | keep `per_device_train_batch_size * gradient_accumulation_steps * n_gpus` a multiple of `num_generations` |
| GRPO rewards stay at -1 | start from an SFT adapter (`model.adapter`); check `log_completions: true` output |
| `trust_remote_code` / gated model errors | use the default Qwen model or log in with `hf auth login` after accepting the license |
| PPO slow | raise `n_envs` towards the core count; GPU does not help a 64x64 MLP |
| A job died | Optuna studies resume with the same `--run-name`. SFT/GRPO keep the last two checkpoints in `runs/<kind>/<run>/checkpoints/`; the simplest recovery is to re-run the command |
| Dataset checksum mismatch | `python -m training.build_datasets` regenerates them deterministically; the manifest must then match again |
