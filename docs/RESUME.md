# RESUME HERE - project paused 2026-10-10

Paused by the owner to work on a higher-priority project. Everything needed to continue is in git (branch **`v3`**
on https://github.com/chetangadhiya5062/open-env-nuclei). The task list is [BRIEF_2.md](BRIEF_2.md) (a copy of
`CLAUDE_CODE_PROMPT_2.md`); design notes and interview explanations are in [PHASE_NOTES.md](PHASE_NOTES.md).

## To continue on any PC (PowerShell)
```powershell
git clone https://github.com/chetangadhiya5062/open-env-nuclei.git
cd open-env-nuclei
git checkout v3
python -m venv .venv                      # or: uv venv --python 3.11 .venv   (Python 3.10+)
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"                   # includes torch + stable-baselines3 for RL training
copy .env.example .env                    # then put your own HF_TOKEN in .env (never commit it)
pytest ; ruff check . ; ruff format --check .      # expect: all green (see "State" below)
```
Then tell Claude Code: "Read docs/RESUME.md and docs/BRIEF_2.md and continue from the first unfinished item."

## State at the pause
Branch `v3` was created from `main` (f645045, PR #8 merged). `main` CI is green. Nothing from `v3` is merged or has a PR.

| Brief part | Status |
|---|---|
| A - sync, CI check, branch list | **Done.** Only open item: `origin/v2` (fully merged) is not deleted - the tool guard blocked remote deletion. Run `git push origin --delete v2` yourself if you want it gone. Other remote branches (`chetan`, `dev`, `sahaj`, `feature/improve-agent`) are also fully merged; owner decides. |
| B - LLM benchmark | **Blocked on you.** Every model on the HF router returns 402 "no remaining credits" (or 400 not supported). Add Inference Providers credits, or point `API_BASE_URL`/`MODEL_NAME` at another OpenAI-compatible endpoint (e.g. local Ollama) in `.env`. Harness is ready: `python benchmark.py --agents llm --seeds 5 --out docs/benchmarks/llm_run`. Details in PHASE_NOTES (Part B). |
| C - improvement metric, harder easy task | **Done, tested** (do-nothing: easy 0.854, medium 0.795, hard 0.608). Full benchmark re-run + README table refresh still to do (do it once, after RL, so one consistent run). |
| D - RL (PPO) | **Code done and tested, final training NOT finished.** See below. |
| E - deploy Space | **Script ready, not run.** `scripts/deploy_space.py` (dry run works). Needs you: `hf auth login` (paste a WRITE token at that prompt), then `python scripts/deploy_space.py`, then check `/health` and `/demo` on https://chetangadhiya017-data-cleaning-env.hf.space. |
| F - wrap up | Not started: README (final table with raw score + improvement, all agents, learning curve, limitations, live link), resume bullet + pitch, push, PR. |

### RL details (Part D)
- Code: `agents/rl_common.py` (54 features, 19 curated actions incl. `finish`), `agents/rl_env.py` (Gymnasium wrapper),
  `agents/rl_agent.py` (numpy-only inference, no torch needed on the Space), `train_rl.py`, tests in `tests/test_rl.py`.
- Seeds: training uses env seeds >= 100, checkpoint selection uses validation seeds 50-59, benchmark/test uses seeds 0-9.
- A full run was started (`python train_rl.py --timesteps 400000 --seed 0`, 8 parallel envs, ~22 min on a 16-core PC) and
  **stopped by hand at ~287k steps**. Interim validation scores (easy / medium / hard, mean grader score on seeds 50-59, greedy policy):
  t=8k untrained 0.888/0.830/0.608 -> t=82k 0.964/0.957/0.857 -> t=184k 0.964/0.957/0.937 -> t=246k **0.967/0.957/0.962**
  (mean improvement 0.82) - clearly learning. For reference the rule-based agent scores about 0.985/0.977/0.97 on test seeds.
  These are interim numbers from a stopped run: **do not put them in the README**; re-run to completion.
- No model file is committed on purpose (the local one was a mid-run checkpoint). To finish:
  ```powershell
  python train_rl.py --timesteps 400000 --seed 0     # writes models/ppo_policy.npz + .json and docs/benchmarks/learning_curve.png
  python benchmark.py --agents do-nothing,random,rule,rl --seeds 10 --out docs/benchmarks   # add llm once Part B works
  ```
  Commit `models/ppo_policy.npz`, `models/ppo_policy.json` (small) and the benchmark/learning-curve files.
  Honest reporting rule from the brief: if PPO does not beat rule-based, say so and describe what it learned.

## Next steps, in order
1. Finish Part D: run training to the end, benchmark all agents on the held-out test seeds 0-9, commit model + results.
2. Part E: log in to HF yourself, run the deploy script, verify the live Space.
3. Part B when credits/endpoint exist: LLM run, describe its typical mistakes from `trace.jsonl`.
4. Part F: README refresh, update resume bullet/pitch in PHASE_NOTES (append sections for Parts D-F, keep existing text), push `v3`, open the PR to `main` (owner reviews and merges).

## Decisions already made (also in PHASE_NOTES)
- Default LLM model is now `meta-llama/Llama-3.1-8B-Instruct` (the router no longer serves Llama 3 8B).
- RL policy is exported to numpy so the Space needs no torch; torch / stable-baselines3 are in the optional `[rl]` extra.
- The `standardize_aliases:city` RL action is a macro that reuses the rule-based alias matcher (stated honestly in the docs).
