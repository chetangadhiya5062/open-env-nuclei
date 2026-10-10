# Brief 2 - finish, deploy, and make it learn

Context: this is my project "open-env-nuclei" (https://github.com/chetangadhiya5062/open-env-nuclei). Phases 1-5 of
../CLAUDE_CODE_PROMPT.md are DONE and merged to `main` via PR #8. Read docs/PHASE_NOTES.md and README.md first so you
know what exists. An independent review confirmed: 79 tests pass, ruff clean, benchmark numbers reproduce exactly.

I'm on Windows, PowerShell. I'm a student; after every part, add a short "what changed, why, and how I'd explain it
in an interview" section to docs/PHASE_NOTES.md (append, keep the existing content).

## Ground rules (same as before)
- Never commit directly to main. Do all new work on a new branch `v3` from the updated `main`, small conventional
  commits, and open a PR to main at the end. I will review and merge it myself.
- Never print, log, commit or echo secrets. HF_TOKEN lives only in `.env` (git-ignored) - read it from there via
  python-dotenv. If `.env` is missing or HF_TOKEN is empty, STOP and ask me to create it; do not ask me to paste the
  token into the chat.
- No invented numbers. Every number in README/notes must come from a run you actually did, and say how it was run.
- Keep everything that already works working: run `pytest` and `ruff check . ; ruff format --check .` before every
  commit. Keep root `inference.py` and its [START]/[STEP]/[END] format, port 7860, the OpenEnv `create_app` server.
- Use the existing `.venv` (`.\.venv\Scripts\python.exe`). Install new deps into it and add them to pyproject.toml
  (and requirements.txt if runtime) - the hygiene test checks they agree.
- Only stop and ask me when you truly need me (login, a secret, a destructive action on GitHub, or an irreversible
  choice). Otherwise make the safest choice, write it under "Decisions I made for you" in PHASE_NOTES.md, and continue.

## Part A - Sync and housekeeping
1. `git checkout main` and `git pull`. Confirm local main == origin/main. Then create branch `v3`.
2. GitHub CI: check the latest Actions runs for main (use `gh run list` / `gh run view --log-failed` if `gh` is
   installed and logged in; otherwise tell me exactly what to click and check). If anything is red, fix it on `v3`.
3. Branch cleanup: list remote branches (`chetan`, `dev`, `sahaj`, `feature/improve-agent`, `v2`) and for each say
   whether it is fully merged into main (`git branch -r --merged origin/main`) and when it last changed. Delete
   `origin/v2` and local `v2` if fully merged. For the others, which belong to teammates Hil and Sahaj too, show me
   the list and ASK me before deleting anything.

## Part B - LLM benchmark (the missing number)
1. Check `.env` has HF_TOKEN (without printing it). If not, stop and ask me.
2. Run: `python benchmark.py --agents do-nothing,random,rule,llm --seeds 5 --out docs/benchmarks/llm_run`.
   If the default model (meta-llama/Meta-Llama-3-8B-Instruct on the HF router) is unavailable or rate-limited, try
   once more, then try one other free instruct model available on the HF router, and record which model was used.
   Handle API errors gracefully (don't let one failed call crash the whole benchmark - count it and report it).
3. Put the real LLM numbers into the README benchmark table (model name, seeds, date) and explain the results
   honestly - including if the LLM is below the rule-based agent, and why (rule-based was written with knowledge of
   the generator; the LLM is zero-shot). Look at the episode logs and describe the LLM's typical mistakes.

## Part C - Make the scores more meaningful
The easy task is too easy: the do-nothing agent already scores 0.954, so results bunch near the top.
1. Add a normalised metric next to the raw grader score: `improvement = (score - do_nothing_score) /
   (1 - do_nothing_score)` for the same task+seed (i.e. "share of the possible improvement achieved"; can be negative
   if the agent made things worse). Report it in the benchmark table, the final observation's score breakdown, and
   the Gradio UI. Add tests.
2. Make `easy-clean` a bit harder so do-nothing scores clearly lower (aim roughly 0.80-0.88), keeping it the easiest
   task. Keep everything deterministic, update openenv.yaml task descriptions so they match what each task really
   contains, update tests.
3. Re-run the full benchmark (all agents incl. llm, 5-10 seeds) and refresh README + docs/benchmarks.

## Part D - Make it actually learn (RL training)
Goal: the README can honestly say an agent *learns*. Keep it CPU-friendly (must train on my laptop in under ~30 min).
1. Add a Gymnasium wrapper around the environment (in-process, no HTTP): featurise the observation into a fixed-size
   vector (per-column issue counts by type, missing, quality score, steps remaining, duplicates) and use a Discrete
   action space over a curated list of (action, column, args) combos, including finish. Use the existing shaped reward.
2. Train with stable-baselines3 PPO (and/or a simple tabular/DQN baseline if it helps), on random seeds per episode,
   evaluating on HELD-OUT seeds that were never used in training. Save the model to `models/` (small file) with a
   training script `train_rl.py`, fixed seeds, and a learning-curve plot (`docs/benchmarks/learning_curve.png`).
3. Add the trained policy as an agent (`rl`) in agents/ and benchmark.py, and to the Gradio demo if the model file
   exists. Compare it to do-nothing, random, rule-based and LLM on held-out seeds, all three tasks.
4. Report honestly: if PPO doesn't beat rule-based, say so and explain what it did learn (e.g. far better than random,
   learned to finish, learned ordering like cast before fill). Add tests (wrapper shapes, action mapping, a tiny
   smoke-train of a few hundred steps). Mark training deps as an optional extra (`[rl]`) so the HF Space stays light,
   unless the Space needs them to run the `rl` agent - then keep the inference part lightweight.

## Part E - Deploy the Hugging Face Space
The live Space (https://chetangadhiya017-data-cleaning-env.hf.space, repo `chetangadhiya017/data-cleaning-env`)
still runs pre-v2 code.
1. Prepare the deploy: the Space needs the YAML card from data_cleaning_env/README.md as the Space's own README.md
   (front-matter at the top), plus the Dockerfile and code. Write a small script `scripts/deploy_space.ps1` (or
   Python using huggingface_hub `upload_folder`) that builds a clean deploy folder with the Space README and uploads
   it, excluding .venv, caches, tests, .env.
2. If I'm not logged in to Hugging Face (`huggingface-cli whoami` / `hf auth whoami`), STOP and tell me the exact
   command to log in myself (I'll paste my write token into that prompt, not into the chat). Then deploy.
3. After deploy, wait for the build and check `/health` and `/demo` on the live URL. If the build fails, read the
   Space build logs and fix. Update README links/notes once it's live.

## Part F - Wrap up
1. Update README: final benchmark table (raw score + improvement, all agents incl. llm and rl), learning curve,
   honest limitations, roadmap, live demo link, contributors kept (Chetan Gadhiya, Hil Kalathiya, Sahaj Saliya).
2. Update the resume bullet and 60-second interview pitch in PHASE_NOTES.md to match what is now true and measured
   (now it may say "trained a PPO agent" if Part D was done).
3. Push `v3`, open the PR to main with a clear description (what changed, the benchmark table, how to verify), and
   give me a short final summary: what's done, what needs me, and the PR link.

Start with Part A now.
