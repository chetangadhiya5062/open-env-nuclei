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
