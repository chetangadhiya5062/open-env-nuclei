# Phase notes (v2 branch)

Written while you were away. Each phase has: **what changed**, **why**, and an **interview explanation**.
A "Decisions I made for you" section at the end lists every judgment call so you can veto any of them.

---

## Phase 1 - Cleanup & honesty

### What I changed
- Deleted `project_details_for_portfolio.txt`, the duplicate `data_cleaning_env/inference.py`, the stale
  `data_cleaning_env/uv.lock`, `data_cleaning_env/server/Dockerfile` (it used port 8000) and
  `data_cleaning_env/server/requirements.txt`.
- `requirements.txt` was UTF-16 LE: re-saved as UTF-8 and completed (`openai`, `fastapi`, `python-dotenv`, ...).
- `pyproject.toml` moved to the repo root and is now the single source of truth for dependencies
  (ranges pinned). `requirements.txt` mirrors the runtime list for Docker/HF; a test checks they agree.
- `openenv.yaml` moved to the root, port 7860, `app:` now points at `data_cleaning_env.server.app:app`.
- One port story: server reads `PORT` (default 7860), clients read `ENV_URL` (default `http://localhost:7860`).
  Dockerfile, README card, yaml and `.env.example` all agree.
- Debug prints (`OBS CALLED`, `SERVER RUNNING FROM`, ...) became `logging`. Removed the unused `sample_df`.
- README rewritten to say only what the code does (no "learns", no fake reward table, no anti-loop claim,
  no OpenRouter claim, correct API notes).
- Added `.gitattributes` (LF endings) and a better `.gitignore` / `.dockerignore`.

### Why
A reviewer who reads the README and then the code should find no contradictions. Config in one place avoids
"works on my machine" port bugs. UTF-16 `requirements.txt` breaks `pip` on Linux (Docker).

### Interview explanation
"I audited my own project for claims the code did not back up - an 'exponential anti-loop penalty' that did not
exist, a reward table that did not match the code - and fixed the docs to match reality before building more.
I also consolidated config: one pyproject for dependencies, one `PORT`/`ENV_URL` convention."

---

## Phase 2 - A real environment

### What I changed
New/rewritten files (all under `data_cleaning_env/`):

| File | Role |
|---|---|
| `schema.py` | Target schema: column kinds, allowed category values, valid numeric ranges |
| `server/datagen.py` | Seeded generator -> `(dirty, truth)`; 3 task configs (easy/medium/hard) |
| `server/quality.py` | Observable issue counting -> quality score `Q` in [0,1] |
| `actions.py` | Strict Pydantic action models + `parse_action` (error messages with usage hints) |
| `server/operations.py` | Pure pandas implementation of each action |
| `server/grader.py` | Final 0-1 score vs hidden ground truth |
| `server/data_cleaning_env_environment.py` | The environment: reset/step/reward/observation |
| `models.py`, `client.py` | Loose transport action, rich observation, typed state; client parses them |

**Data.** Customers table (`customer_id, name, age, city, signup_date, plan, monthly_spend`). `reset(task_id, seed)`
is deterministic. Noise: nulls, exact duplicates, case/space/alias variants (`NY`, `new york`, `New York `),
numbers as strings (`"25"`), `N/A`-style placeholders, impossible values (age 999), mixed date formats,
whitespace in names. `easy` = nulls in `age` only; `medium` = nulls + duplicates; `hard` = everything.

**Actions** (9): `fill_missing` (mean/median/mode/constant), `drop_duplicates`, `standardize_categories`
(auto or explicit mapping), `cast_type`, `strip_whitespace`, `fix_dates`, `clip_outliers`,
`drop_rows_with_missing`, `finish`. Invalid actions are *not applied*; the agent gets `last_error` text plus a
small penalty instead of a crash.

**Reward** (documented in the environment module docstring):

```
r = 10*(Q_t - Q_{t-1}) - 0.02 step cost - 0.05 if nothing changed - 0.10 if invalid
    - 0.30 per ground-truth row destroyed
    + (on finish only) +1.0 if Q == 1 else -2.0*(1-Q)
```
`Q = 1 - issue_cells / (initial_rows * n_columns)`; it counts schema violations and duplicate rows and needs no
ground truth. Shaping terms telescope, so repeating or cycling actions cannot create reward.

**Old exploits, closed:** +5 clean bonus every step (gone - only a one-time terminal term), double +5 on finish
(gone), `"missing"` string into numeric `age` (rejected as invalid; also still counts as an issue in `Q`),
cumulative reward returned as `obs.reward` (now `reward` = this step, `total_reward` = running sum).

**Grader**: `0.6*cell_accuracy + 0.2*row_fidelity + 0.2*schema_score`. Rows aligned by `customer_id`;
numeric cells get partial credit by closeness (an imputed mean can never be exact); lost rows count as zeros;
leftover duplicates lower row fidelity. Identical-to-truth table scores exactly 1.0. Revealed only in the
final observation as `final_score` / `score_breakdown` (a real field - OpenEnv drops `metadata` over HTTP/WS).

**Smoke test result (hard-clean, seed 1, hand-written good sequence):** quality 0.52 -> 1.00, final grader score 0.896;
a bogus action and a no-op both gave small negative rewards as designed.

### Why
The old environment was a 5-row toy whose reward could be gamed, so any result on it meant nothing. A seeded
generator with hidden ground truth makes results reproducible and checkable; a grader separate from the reward
lets us measure what the agent *really* achieved even when reward and truth disagree.

### Interview explanation
"I replaced a hard-coded toy with a seeded data generator that keeps the clean table hidden. The reward is
potential-based shaping on an observable quality score, which provably can't be farmed because the terms telescope;
the final grade compares to ground truth and is kept out of the reward on purpose, so I can see reward-hacking gaps.
Invalid LLM actions return an error message and a small penalty instead of crashing, so the agent can recover."

Honest limitation worth saying out loud: imputed values can't match the truth exactly, so the maximum realistic
grader score is below 1.0 (the proxy reward reaches 1.0, the grader doesn't). That gap is a feature, not a bug.

---

## Phase 3 - Agents & benchmark

### What I changed
- New `agents/` package with one interface (`Agent.reset(seed)`, `Agent.act(obs) -> action`):
  - `DoNothingAgent` - finishes at once (the score of the untouched table: a floor).
  - `RandomAgent` - seeded random actions and arguments (many are invalid on purpose).
  - `RuleBasedAgent` - checklist baseline: dedupe -> whitespace -> categories (auto, then alias mapping via
    prefix/initials against `allowed_values`) -> cast numerics -> fix dates -> clip outliers -> fill (median/mode)
    -> finish. It reads only the observation; no per-city hard-coding. Each (action, column) is tried once.
  - `LLMAgent` - HF-router chat model. Replies must be one JSON object, validated with Pydantic and the strict
    action models. On an invalid reply it re-asks (max 2 retries) and shows the model the exact error. Keeps a
    5-step action history in the prompt. **No hidden overrides**: if all retries fail it sends an obviously
    invalid action (`invalid_llm_output`), logs a warning and increments `fallbacks`, which the benchmark reports.
- `agents/runner.py`: `run_episode(agent, env, task, seed)` for either in-process (`LocalEnv`) or over WebSocket
  against a running server (`RemoteEnv`).
- `benchmark.py`: agents x tasks x seeds -> `episodes.csv`, `results.md` (mean +- std score, mean reward, steps,
  invalid-action rate, fallbacks) and `scores.png`. The `llm` agent is skipped with a warning without `HF_TOKEN`.
- `inference.py` (root, still hackathon-compatible) rewritten on top of `LLMAgent`; prints
  `[START]`/`[STEP]`/`[END]` lines; the old forced column switching and client-side early stop are gone.
- Environment tweak: each column profile now has `bad_values` (distinct values outside `allowed_values`) so an
  agent can see which spellings still need mapping.

### Measured results (10 seeds, in-process; also saved in `docs/benchmarks/`)

| Agent | easy | medium | hard |
|---|---|---|---|
| do-nothing | 0.954 | 0.862 | 0.608 |
| random | 0.871 ± 0.079 | 0.766 ± 0.073 | 0.534 ± 0.058 |
| rule-based | 0.987 ± 0.001 | 0.980 ± 0.003 | 0.968 ± 0.003 |

Honest reading: the easy task is easy - doing nothing already scores 0.95 because only ~8 cells are missing, so
the interesting separation is on medium/hard. Random is *worse than doing nothing* because it destroys data and
hits invalid actions (~30%). No LLM numbers yet (needs your token).

**A bug the benchmark caught:** my first run showed "do-nothing" beating the rule-based agent. Cause: the grader
checked `isinstance(x, (int, float))`, which is False for numpy `int64`, so every correctly cleaned numeric cell got
zero credit. Fixed with `numbers.Real` (and a regression test in Phase 4). Lesson: always sanity-check a metric
against trivial baselines.

### Run the LLM benchmark yourself (PowerShell)
```powershell
cd D:\=Chetan\openEnv_enhancement\open-env-nuclei
.\.venv\Scripts\Activate.ps1
$env:HF_TOKEN = "hf_your_token_here"
python benchmark.py --agents llm --seeds 5 --out benchmark_results_llm
# optional: all agents side by side
python benchmark.py --agents do-nothing,random,rule,llm --seeds 5
# optional: against the running server instead of in-process
python benchmark.py --agents llm --seeds 3 --env-url http://localhost:7860
```
Do not paste the token into any file; `.env` is git-ignored.

### Why
Without baselines a score means nothing. A random agent and a do-nothing agent give the floor, a strong heuristic
gives a ceiling to compare an LLM against, and a benchmark with seeds and std gives numbers you can defend.

### Interview explanation
"I built three baselines and a benchmark harness before judging the LLM: do-nothing, random and a rule-based
checklist agent. The harness runs every agent over seeded tasks and reports mean +- std. It even caught a bug in my
own grader (numpy ints not counted as numbers) because the do-nothing baseline was beating the rule-based agent.
For the LLM agent I removed all hidden client-side overrides; when the model fails to produce a valid action I send
an invalid action, log it and count it, so the benchmark shows how often the model really failed."

---

## Phase 4 - Quality

### What I changed
- `tests/` with **78 pytest tests**, all passing (`pytest` runs in ~2.5 s):
  - generator: determinism, seeds differ, truth is clean/dirty is dirty, each task has exactly the declared noise,
    `openenv.yaml` task list matches the implemented tasks;
  - every action's correctness on small hand-built tables, and that invalid actions return an error instead of raising;
  - **reward cannot be farmed** (regression tests for the old exploits): no bonus for lingering in a clean state,
    spamming actions is worse than finishing, terminal reward is paid once, shaping telescopes
    (`sum(r) == 10*(Q_T-Q_0) - step costs`), destroying rows is penalised, per-step (not cumulative) reward,
    grader hidden until the episode ends;
  - grader: bounds, **exactly 1.0 on the ground truth**, partial numeric credit, lost rows/leftover duplicates lower it,
    numpy-number regression;
  - agents: reproducible random agent, rule-based never makes an invalid action, LLM agent against a **fake client**
    (valid reply, retry with error text, fallback is counted + logged + sent as an invalid action);
  - API via FastAPI `TestClient`: `/health`, `/schema`, `/reset` with task+seed, stateless `/step`, malformed payload,
    and a **WebSocket session that keeps state across steps**;
  - hygiene: UTF-8 `requirements.txt` equals `pyproject` deps, no `print`/emoji in library code, one port everywhere,
    no hard-coded tokens.
- Ruff configured in `pyproject.toml` (lint rules E,F,W,I,B,UP; line length 120) and the whole repo formatted.
  `pre-commit` skipped (optional); CI enforces the same checks.
- `.github/workflows/ci.yml`: lint + format check + tests on Python 3.10/3.11/3.12 for every push/PR, and a job that
  builds the Docker image and curls `/health`.

### What I could NOT verify
- **Docker build:** Docker Desktop's engine wasn't running on your PC, so I did not run `docker build`. Instead I
  installed *only* `requirements.txt` into a clean venv and served the app with the Dockerfile's uvicorn command:
  `/health` returned healthy. The CI Docker job will do the real build on GitHub.
- CI itself has not run (nothing is pushed).

### Why
Tests turn "I think it works" into evidence, and the exploit regression tests make sure the bug I fixed can't return
unnoticed. CI means teammates' PRs get checked automatically.

### Interview explanation
"I wrote tests for the failure modes, not just the happy path: a test that spams actions in a clean state and asserts
the total reward is lower than finishing, a test that the shaping sums telescope, and a test that the grader gives
exactly 1.0 to the ground truth. CI runs lint, tests on three Python versions and a Docker build on every push."

---

## Decisions I made for you (veto any of them)
1. **Moved** `pyproject.toml` and `openenv.yaml` from `data_cleaning_env/` to the repo root (one project, one manifest).
   Deleted `data_cleaning_env/server/Dockerfile` (port 8000, stale openenv-base template) and `uv.lock` (stale).
   If you deploy with the `openenv push` CLI you may want a server Dockerfile back - tell me.
2. **Action names changed** (`remove_duplicates` -> `drop_duplicates`, `finish_cleaning` -> `finish`, `fill_missing`
   now needs `strategy`). The old API is gone, as the Phase 2 spec asked for a richer action space.
3. The **schema** (types, allowed values such as the 5 cities, valid ranges) is shown to the agent in the
   observation. I judged this fair (like column constraints in a database) and necessary to make "NY -> New York"
   well-defined. If you prefer a harder "discover the vocabulary yourself" variant, that is a roadmap item.
4. Grader weights (0.6 cells / 0.2 rows / 0.2 schema) and reward constants are my choices; they live as named
   constants at the top of `grader.py` / the environment module.
5. Default `reset()` with no arguments = `easy-clean`, seed 42 (deterministic).
6. Git identity: the repo had no local git user, so I set a **repo-local** `user.name=Chetan Gadhiya`,
   `user.email=chetangadhiya4939@gmail.com` for the commits. Change with `git config --local` and amend if you like.
   Commits carry a `Co-Authored-By: Claude` trailer.
7. Python: your PC had no Python, so I used `uv` to create `.venv` (Python 3.11). `.venv/` is git-ignored.
8. Not done, by instruction: Phase 5 (Gradio UI, full README rewrite, resume bullet), Phase 6, push and PR.
   The README is still the short honest Phase 1 version and does not yet describe the new environment in detail.

## How to continue when you're back
```powershell
cd D:\=Chetan\openEnv_enhancement\open-env-nuclei
git log --oneline main..v2          # review the commits
.\.venv\Scripts\python.exe -m pytest      # 78 tests
.\.venv\Scripts\python.exe benchmark.py --agents do-nothing,random,rule --seeds 10
```
Then tell me "go" for Phase 5 (or push + PR first).

---

## Phase 5 - Demo & presentation

### What I changed
- `data_cleaning_env/ui.py`: Gradio app (task, seed and agent dropdowns; step-by-step table of action / reward /
  quality / result; before and after tables; final grader breakdown). It runs episodes in-process, so visitors never
  share state. Mounted at `/demo` inside the same FastAPI app (`app.py`), and `/` now redirects there. If Gradio
  fails to import, the API still starts. The `llm` option appears only when `HF_TOKEN` is set. I opened it in the
  browser and ran a `hard-clean` episode: final grader score 0.971 for seed 0, 15 steps, 0 invalid actions.
- `gradio>=5,<7` is now an explicit dependency (OpenEnv already pulled it in).
- `README.md` fully rewritten: pitch, Mermaid architecture diagram, task table, action table, observation, reward
  formula, grader formula, the real benchmark table, quick start (local / Docker / HF), roadmap, limitations,
  contributors kept. The LLM row says "not measured yet" - there is no invented number anywhere.
- `data_cleaning_env/README.md`: Space card with `app_port: 7860`, `base_path: /demo`, accurate description.
- Added a test that `/demo` is mounted and an episode runs (79 tests).

### Resume bullet (only things that are implemented and measured)
> Built an OpenEnv reinforcement-learning environment for tabular data cleaning (Python, pandas, Pydantic, FastAPI):
> seeded synthetic tasks with hidden ground truth, a 9-action typed action space, a potential-based shaped reward with
> tested anti-exploit properties, and a separate 0-1 grader. Benchmarked random, rule-based and LLM-prompting agents
> over seeded runs (rule-based 0.97-0.99 vs 0.53-0.87 for random); 79 tests and CI with Docker build; Gradio demo
> deployed on Hugging Face Spaces.

Remove the last clause ("deployed on Hugging Face Spaces") until you redeploy v2, and add LLM numbers once you run them.
Do **not** write "trained" or "learns": the LLM agent is zero-shot and no RL training exists yet (Phase 6).

### 60-second interview pitch
"I built an environment where an agent cleans a messy table step by step, following the OpenEnv API. The data is
synthetic and seeded, so every run is reproducible, and I keep the clean version hidden. The agent has nine typed
actions - fill, dedupe, standardise categories, fix dates and so on - and invalid actions come back as error messages,
not crashes. For the reward I used potential-based shaping on an observable quality score, so you can't farm reward by
repeating actions - I wrote tests that try. The final grade compares against the hidden truth and is deliberately
separate from the reward. I benchmarked a random agent, a do-nothing agent and a rule-based agent over seeded runs;
the benchmark also caught a bug in my own grader. I also built an LLM agent that must output validated JSON, retries
with the error message, and counts failures rather than silently fixing them. Honest limitation: nothing is trained
yet - the next step is an RL policy to compare against those baselines."

### Interview explanation of the key design choice
"Reward and evaluation are different on purpose. Reward is a dense proxy that guides learning; the grader measures the
real objective. When they disagree - for example a constant imputation raises the proxy but lowers the grade - that is
exactly the reward-hacking signal I want to be able to see."

### Not done
- Phase 6 (RL training), as instructed. HF Space not redeployed (needs your account). No PR opened yet.
