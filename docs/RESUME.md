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

## State

| Brief part | Status |
|---|---|
| A - sync, CI check, branch list | **Done.** Only open item: `origin/v2` (fully merged) is not deleted - the tool guard blocked remote deletion. Run `git push origin --delete v2` yourself if you want it gone. Other remote branches (`chetan`, `dev`, `sahaj`, `feature/improve-agent`) are also fully merged; owner decides. |
| B - LLM benchmark | **Blocked on you.** Every model on the HF router returns 402 "no remaining credits" (or 400 not supported). Add Inference Providers credits, or point `API_BASE_URL`/`MODEL_NAME` at another OpenAI-compatible endpoint (e.g. local Ollama) in `.env`. Harness is ready: `python benchmark.py --agents llm --seeds 5 --out docs/benchmarks/llm_run`. Details in PHASE_NOTES (Part B). |
| C - improvement metric, harder easy task | **Done, tested** (do-nothing: easy 0.854, medium 0.795, hard 0.608). |
| D - RL (PPO) | **Done, trained & benchmarked.** 400k steps completed, best validation mean improvement 0.823 at $t=401,408$. Exported to `models/ppo_policy.npz`, benchmarked on held-out test seeds 0–9: easy 0.964 (+0.755), medium 0.960 (+0.806), hard 0.967 (+0.917) with 0.0% invalid-action rate. |
| E - deploy Space | **Script ready, not run.** `scripts/deploy_space.py` (dry run works). Needs you: `hf auth login` (paste a WRITE token at that prompt), then `python scripts/deploy_space.py`, then check `/health` and `/demo` on https://chetangadhiya017-data-cleaning-env.hf.space. |
| F - wrap up | **Done.** README updated with final benchmark table, learning curve, honest limitations, and updated roadmap. PHASE_NOTES updated with Part D–F sections, updated resume bullet, and 60-second interview pitch. |

### RL details (Part D)
- Code: `agents/rl_common.py` (54 features, 19 curated actions incl. `finish`), `agents/rl_env.py` (Gymnasium wrapper),
  `agents/rl_agent.py` (numpy-only inference, no torch needed on the Space), `train_rl.py`, tests in `tests/test_rl.py`.
- Seeds: training uses env seeds >= 100, checkpoint selection uses validation seeds 50-59, benchmark/test uses held-out seeds 0-9.
- A full 400k run finished cleanly with 8 parallel workers. Validation scores progressed from untrained baseline 0.888/0.830/0.608 -> best 0.964/0.959/0.968 (mean improvement 0.823).
- Test benchmark (seeds 0–9):
  - easy: 0.964 ± 0.006 (+0.755)
  - medium: 0.960 ± 0.004 (+0.806)
  - hard: 0.967 ± 0.004 (+0.917)
  - invalid-action rate: 0.0%
- Artifacts: `models/ppo_policy.npz`, `models/ppo_policy.json`, `docs/benchmarks/learning_curve.png`, `docs/benchmarks/scores.png`, `docs/benchmarks/episodes.csv`, `docs/benchmarks/results.md`.

## Next steps, in order
1. Part E: Log in to Hugging Face yourself (`hf auth login`), run `python scripts/deploy_space.py`, and verify https://chetangadhiya017-data-cleaning-env.hf.space.
2. Part B (when credits/endpoint exist): `python benchmark.py --agents llm --seeds 5`, analyze `trace.jsonl`, and record numbers in README.
3. Review branch `v3` and merge PR into `main`.

## Decisions already made (also in PHASE_NOTES)
- Default LLM model is now `meta-llama/Llama-3.1-8B-Instruct` (the router no longer serves Llama 3 8B).
- RL policy is exported to numpy so the Space needs no torch; torch / stable-baselines3 are in the optional `[rl]` extra.
- The `standardize_aliases:city` RL action is a macro that reuses the rule-based alias matcher (stated honestly in the docs).
