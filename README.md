# OpenEnv Data Cleaning Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an agent cleans a dirty pandas table one
action at a time, plus baselines, a benchmark, an LLM agent and a Gradio demo.

- **Reproducible:** every episode is `(task_id, seed)` -> the same dirty table, with a hidden clean ground truth.
- **Honest scoring:** the training-style *reward* and the final *grader score* are separate things (see below).
- **Agents included:** do-nothing, random, a rule-based heuristic, a trained RL agent (PPO), and a zero-shot LLM agent (Llama 3.1 8B through the Hugging Face router). The RL agent is trained with PPO on held-out seeds and runs via lightweight pure-NumPy inference with zero PyTorch dependencies at runtime.

> The Hugging Face Space demo runs at [chetangadhiya017-data-cleaning-env.hf.space](https://chetangadhiya017-data-cleaning-env.hf.space). To redeploy the latest code, see [Deploy](#deploy-to-hugging-face-spaces).

## How it works

```mermaid
flowchart LR
    A[Agent<br/>rule-based / random / RL / LLM] -- action JSON --> S
    subgraph S[OpenEnv server  FastAPI + WebSocket]
      V[Pydantic validation<br/>actions.py] --> O[pandas operations<br/>operations.py]
      O --> Q[quality score Q<br/>quality.py]
      Q --> R[reward]
      G[(hidden ground truth<br/>datagen.py)] --> GR[grader.py]
    end
    S -- observation + reward --> A
    GR -. final_score only when done .-> A
```

Episode: `reset(task_id, seed)` -> repeat `step(action)` -> `finish` (or the step budget runs out).

### Tasks

| Task | Rows | Step budget | What is wrong with the data |
|---|---|---|---|
| `easy-clean` | 40 | 15 | missing values in `age`, `monthly_spend`, and `plan` |
| `medium-clean` | 60 (+duplicates) | 25 | missing values in four columns and 15% exact duplicate rows |
| `hard-clean` | 100 (+duplicates) | 45 | all of the above plus inconsistent categories (`NY`, `new york`, `New York `), numbers stored as text (`"25"`), `N/A` placeholders, impossible values (age 999), mixed date formats, stray whitespace |

The table is a synthetic customers table: `customer_id, name, age, city, signup_date, plan, monthly_spend`.

### Actions

| `action_type` | Fields | Effect |
|---|---|---|
| `fill_missing` | `column_name`, `strategy` (`mean`/`median`/`mode`/`constant`), `value` (constant only) | fill nulls |
| `drop_duplicates` | - | remove exact duplicate rows |
| `standardize_categories` | `column_name`, optional `mapping` | fix case/spacing; map aliases like `NY -> New York` |
| `cast_type` | `column_name`, `target_type` (`int`/`float`/`str`) | convert text to numbers (placeholders become nulls) |
| `strip_whitespace` | optional `column_name` | trim and collapse spaces |
| `fix_dates` | `column_name` | normalise mixed formats to `YYYY-MM-DD` |
| `clip_outliers` | `column_name` | clip to the column's valid range |
| `drop_rows_with_missing` | optional `column_name` | drop rows with nulls (penalised: loses data) |
| `finish` | - | end the episode |

Invalid actions (unknown column, `mean` on text, a string constant in a numeric column, ...) are **not applied**; the
agent receives an explanation in `last_error` and a small penalty.

### Observation

`task_id, seed, row_count, duplicate_count, total_missing, step_count, steps_remaining, quality_score, total_reward,
invalid_action_count, last_action, last_action_ok, last_message, last_error, data_sample` (first 5 rows) and one
**column profile** per column: `dtype, expected_type, missing, unique, sample_values, issues` (e.g.
`{"missing": 5, "whitespace": 3}`), `allowed_values`, `bad_values`, `valid_range`. The expected types, allowed
category values and valid ranges are part of the task description, like constraints in a database schema.
`final_score` and `score_breakdown` are `null` until the episode is over.

### Reward

`Q` is an observable quality score: `Q = 1 - (schema violations + duplicate_rows * n_columns) / (initial_rows * n_columns)`.

```
reward = 10 * (Q_now - Q_before)          progress (potential-based shaping)
       - 0.02                             every step
       - 0.05  if the action changed nothing
       - 0.10  if the action was invalid
       - 0.30  per ground-truth row destroyed
       + on `finish` only: +1.0 if Q = 1, otherwise -2.0 * (1 - Q)
```

The progress terms add up to `10 * (Q_end - Q_start)`, so repeating actions or going in circles cannot earn reward
(there is a test for this). `observation.reward` is the reward of the last step only; `total_reward` is the sum.

### Grader (separate from the reward)

`score = 0.6 * cell_accuracy + 0.2 * row_fidelity + 0.2 * schema_score`, in [0, 1], computed against the hidden
ground truth and revealed only in the final observation. Numeric cells earn partial credit by closeness (an imputed
value is never exactly right); a table identical to the ground truth scores exactly 1.0. Because imputation is
approximate, the best realistic score is below 1.0 even when `Q` reaches 1.0.

## Benchmark

Mean grader score ± std and normalized improvement over 10 held-out test seeds (seeds 0–9) per task, in-process, from `python benchmark.py` (raw per-episode data in [docs/benchmarks/](docs/benchmarks/)):

| Agent | easy-clean (score / impr) | medium-clean (score / impr) | hard-clean (score / impr) | Invalid-action rate |
|---|---|---|---|---|
| do-nothing (finish immediately) | 0.854 ± 0.000 (+0.000) | 0.795 ± 0.000 (+0.000) | 0.608 ± 0.000 (+0.000) | 0.0% |
| random | 0.627 ± 0.140 (-1.558) | 0.605 ± 0.120 (-0.930) | 0.534 ± 0.058 (-0.189) | 27–31% |
| rule-based (heuristic checklist) | **0.967 ± 0.006 (+0.774)** | **0.960 ± 0.004 (+0.806)** | **0.968 ± 0.003 (+0.918)** | 0.0% |
| rl-ppo (400k steps, greedy policy) | 0.964 ± 0.006 (+0.755) | **0.960 ± 0.004 (+0.806)** | 0.967 ± 0.004 (+0.917) | 0.0% |
| LLM (Llama 3.1 8B, zero-shot) | *not measured* | *not measured* | *not measured* | - |

> **Normalized Improvement Metric:** `improvement = (score - do_nothing_score) / (1 - do_nothing_score)` for the same task and seed. 1.0 = perfect restoration, 0.0 = no gain over leaving the dirty data untouched, negative = degraded data quality.

![benchmark chart](docs/benchmarks/scores.png)

### RL Training & Learning Curve

The RL agent was trained for 400,000 steps using Stable-Baselines3 PPO across 8 parallel environment workers (`SubprocVecEnv`), evaluated every 20,480 steps on held-out validation seeds 50–59, and benchmarked on held-out test seeds 0–9 (which were never exposed during training). Weights are saved as `models/ppo_policy.npz` (and metadata in `models/ppo_policy.json`).

![learning curve](docs/benchmarks/learning_curve.png)

**Key learning insights:**
- **Zero invalid actions:** The policy learned to never issue invalid actions across any task or test seed (0.0% invalid-action rate).
- **Matched heuristic performance:** On `hard-clean`, PPO learned to reach **0.967 ± 0.004 (+0.917 improvement)**, virtually identical to the hand-crafted rule-based agent (**0.968 ± 0.003 (+0.918 improvement)**).
- **Proper operation sequencing:** PPO learned to strip whitespace and standardise categories before handling missing values, avoided destructive row drops, and called `finish` once high quality was attained.
- **Lightweight deployment:** The policy is exported to NumPy arrays and runs with zero PyTorch dependencies on CPU in the Space demo.

## Quick start (Windows PowerShell)

Python 3.10+ is required.

```powershell
git clone https://github.com/chetangadhiya5062/open-env-nuclei.git
cd open-env-nuclei
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python -m data_cleaning_env.server.app      # server + demo on http://localhost:7860  (port from $env:PORT)
```

- Demo UI: <http://localhost:7860/demo> - pick task, seed and agent and watch the steps and before/after tables.
- API docs: <http://localhost:7860/docs>.

```powershell
pytest                                       # 97 tests
ruff check . ; ruff format --check .         # lint

python benchmark.py                          # do-nothing, random, rule-based, rl over 10 seeds
$env:HF_TOKEN = "hf_..."                     # your token; never commit it (see .env.example)
python inference.py --task hard-clean        # LLM agent against the running server, [START]/[STEP]/[END] output
```

Use the client from Python:

```python
from data_cleaning_env import DataCleaningAction, DataCleaningEnv

with DataCleaningEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task_id="medium-clean", seed=3)
    result = env.step(DataCleaningAction(action_type="drop_duplicates"))
    print(result.reward, result.observation.quality_score)
```

> `/reset` and `/step` over plain HTTP are **stateless** (each call builds a fresh environment). Real multi-step
> episodes use the WebSocket client above (`/ws`).

### Docker

```powershell
docker build -t openenv-data-cleaning .
docker run -p 7860:7860 openenv-data-cleaning
```

### Deploy to Hugging Face Spaces

The deployment script [`scripts/deploy_space.py`](scripts/deploy_space.py) packages the environment, agents, and exported model weights without dev dependencies:

```powershell
python scripts/deploy_space.py --dry-run     # verify package contents
# To deploy to https://huggingface.co/spaces/chetangadhiya017/data-cleaning-env:
# Run 'hf auth login' (paste write token), then:
python scripts/deploy_space.py
```

## Project layout

```
data_cleaning_env/   schema, models, client, actions, ui and server/ (datagen, quality, operations, grader, environment)
agents/              base interface, random, do-nothing, rule-based, RL agent (numpy), LLM agent, episode runner
models/              trained PPO policy weights (ppo_policy.npz) and training metadata
benchmark.py         agents x tasks x seeds -> markdown / CSV / chart
train_rl.py          PPO training pipeline (8 parallel workers, validation evaluation, policy export)
inference.py         LLM agent against a running server (hackathon-style logs)
tests/               pytest suite (97 tests)
docs/PHASE_NOTES.md  design notes and interview explanations
```

## Roadmap

- [x] Train an RL policy (PPO with Stable-Baselines3, exported to NumPy for lightweight inference) and benchmark it against baselines.
- [ ] LLM benchmark numbers in the table above (requires active Hugging Face inference quota or local Ollama endpoint).
- [ ] A harder variant where the allowed vocabulary is not given to the agent.
- [ ] More schemas / real CSV datasets.

## Limitations

- The data is synthetic; results do not transfer to real data automatically.
- `Q` is a proxy for quality: an agent can raise `Q` with a bad imputation (e.g. a constant). The grader exists to
  catch such gaps.
- Only one grader weighting and one noise model are implemented.
- Action-space engineering: the RL agent's `standardize_aliases:city` action is a macro that generates alias mappings using the rule-based heuristic, so the policy learns *when* to execute the transformation rather than discovering arbitrary character mappings from scratch.

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/chetangadhiya5062">
        <img src="https://github.com/chetangadhiya5062.png" width="100px;" alt=""/>
        <br />
        <sub><b>Chetan Gadhiya</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/HilKalathiya">
        <img src="https://github.com/HilKalathiya.png" width="100px;" alt=""/>
        <br />
        <sub><b>Hil Kalathiya</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/SahajIVVIX-1">
        <img src="https://github.com/SahajIVVIX-1.png" width="100px;" alt=""/>
        <br />
        <sub><b>Sahaj Saliya</b></sub>
      </a>
    </td>
  </tr>
</table>

## License

MIT, see [LICENSE](LICENSE).
