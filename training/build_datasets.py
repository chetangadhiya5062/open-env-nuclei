"""Build every training / evaluation dataset deterministically and write ``training/datasets/manifest.json``.

    python -m training.build_datasets                    # full build (configs/datasets.yaml)
    python -m training.build_datasets --smoke            # tiny build into training/runs/smoke_datasets
    python -m training.verify_datasets                   # check sizes + sha256 against the manifest

Outputs (``training/datasets/``):

* ``env_seeds.json``            train / validation / test seed ranges for every environment-based job
* ``sft/train.jsonl.gz``, ``sft/val.jsonl.gz``
      chat examples ``{"prompt": [system, user], "completion": [assistant], ...}``. The prompt is *exactly* what
      ``agents.LLMAgent`` sends (same system prompt, observation formatting and 5-step history), the completion is the
      rule-based expert's action as JSON. Part of the episodes start with 1-3 random actions so the model also learns
      to recover from bad states (DAgger-style coverage, Ross et al. 2011).
* ``grpo/train.jsonl.gz``, ``grpo/val.jsonl.gz``
      prompts plus ``task_id``, ``seed`` and ``prefix`` (actions already applied) so the reward function can rebuild
      the exact environment state and score any generated action with the environment's own reward.
* ``tabular/<name>_{train,holdout}.parquet``
      messy tables (churn / housing / transactions) for the feature-engineering benchmark.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from training.common import DATASETS, RUNS, apply_smoke, base_parser, load_config, save_json, setup, sha256, write_jsonl

logger = logging.getLogger("training.build_datasets")


def _history_line(action_payload: Dict[str, Any], obs) -> str:
    """Same text as ``LLMAgent._record_previous``."""
    status = "ok" if obs.last_action_ok else f"REJECTED ({obs.last_error})"
    return f"- {json.dumps(action_payload)} -> {status}; reward={obs.reward}"


def trajectory(task_id: str, seed: int, n_random: int, rng: np.random.Generator) -> Iterator[Dict[str, Any]]:
    """Yield one record per expert decision of an episode (optionally after ``n_random`` random actions)."""
    from agents import RandomAgent, RuleBasedAgent
    from agents.llm_agent import HISTORY_LENGTH, SYSTEM_PROMPT, format_observation
    from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment

    env = DataCleaningEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    history: List[str] = []
    prefix: List[Dict[str, Any]] = []
    rand = RandomAgent()
    rand.reset(int(rng.integers(1 << 30)))
    for _ in range(n_random):
        if obs.done:
            return
        a = rand.act(obs)
        if a.action_type == "finish":
            continue
        obs = env.step(a)
        prefix.append(a.to_payload())
        history.append(_history_line(a.to_payload(), obs))
    expert = RuleBasedAgent()
    expert.reset(seed)
    step = 0
    while not obs.done:
        a = expert.act(obs)
        payload = a.to_payload()
        user = format_observation(obs, history[-HISTORY_LENGTH:])
        yield {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": json.dumps(payload, separators=(",", ":"))}],
            "task_id": task_id,
            "seed": int(seed),
            "step": step,
            "n_random_prefix": n_random,
            "prefix": json.dumps(prefix),
        }
        obs = env.step(a)
        prefix.append(payload)
        history.append(_history_line(payload, obs))
        step += 1


def build_env_split(
    tasks: List[str], seeds: Tuple[int, int], perturb_prob: float, max_random: int, seed: int
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for s in range(seeds[0], seeds[1]):
            k = int(rng.integers(1, max_random + 1)) if rng.random() < perturb_prob else 0
            rows.extend(trajectory(task, s, k, rng))
    return rows


def grpo_rows(rows: List[Dict[str, Any]], max_rows: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """Prompt-only records (no completion) for GRPO; subsampled to ``max_rows``."""
    out = [{k: v for k, v in r.items() if k != "completion"} for r in rows]
    if max_rows and len(out) > max_rows:
        idx = np.random.default_rng(seed).choice(len(out), max_rows, replace=False)
        out = [out[i] for i in sorted(idx)]
    return out


def build_tabular(cfg: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    from featureforge.io import write_table
    from featureforge.samples import TARGETS, make_dataset

    files = []
    for name, spec in cfg.items():
        for part in ("train", "holdout"):
            df = make_dataset(name, int(spec[f"{part}_rows"]), int(spec[f"{part}_seed"]))
            path = out / "tabular" / f"{name}_{part}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_table(df, path)  # mixed-type columns are stored as text (that is part of the mess)
            files.append(
                {
                    "path": path,
                    "rows": len(df),
                    "kind": "tabular",
                    "description": f"{name} ({part}), target {TARGETS[name]!r}, generator seed {spec[f'{part}_seed']}",
                }
            )
    return files


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = base_parser(__doc__, "datasets.yaml")
    args = ap.parse_args()
    setup()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    out = RUNS / "smoke_datasets" if args.smoke else DATASETS
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = cfg["env"]
    tasks = env_cfg["tasks"]
    splits = env_cfg["seeds"]
    for a, b in (("train", "val"), ("train", "test"), ("val", "test"), ("grpo_train", "test"), ("grpo_val", "test")):
        lo1, hi1 = splits[a]
        lo2, hi2 = splits[b]
        if max(lo1, lo2) < min(hi1, hi2):
            raise SystemExit(f"seed ranges {a} and {b} overlap")
    save_json(out / "env_seeds.json", {"tasks": tasks, **splits, "note": "[start, end) ranges; test = benchmark seeds"})
    files: List[Dict[str, Any]] = [
        {"path": out / "env_seeds.json", "rows": None, "kind": "seeds", "description": "seed splits"}
    ]

    sft = env_cfg["sft"]
    for part in ("train", "val"):
        rows = build_env_split(tasks, tuple(splits[part]), sft["perturb_prob"], sft["max_random_actions"], sft["seed"])
        n = write_jsonl(out / "sft" / f"{part}.jsonl.gz", rows)
        files.append(
            {
                "path": out / "sft" / f"{part}.jsonl.gz",
                "rows": n,
                "kind": "sft",
                "description": f"expert (rule-based) actions, env seeds {splits[part]}",
            }
        )
        logger.info("sft/%s: %d examples", part, n)

    grpo = env_cfg["grpo"]
    for part in ("train", "val"):
        rows = build_env_split(
            tasks, tuple(splits[f"grpo_{part}"]), grpo["perturb_prob"], grpo["max_random_actions"], grpo["seed"]
        )
        rows = grpo_rows(rows, grpo.get(f"max_{part}_prompts"), grpo["seed"])
        n = write_jsonl(out / "grpo" / f"{part}.jsonl.gz", rows)
        files.append(
            {
                "path": out / "grpo" / f"{part}.jsonl.gz",
                "rows": n,
                "kind": "grpo",
                "description": f"prompt states with replayable prefix, env seeds {splits[f'grpo_{part}']}",
            }
        )
        logger.info("grpo/%s: %d prompts", part, n)

    files += build_tabular(cfg["tabular"], out)

    manifest = {
        "created_by": "python -m training.build_datasets" + (" --smoke" if args.smoke else ""),
        "config": cfg,
        "git_commit": git_commit(),
        "files": [
            {
                "path": str(f["path"].relative_to(out)).replace("\\", "/"),
                "bytes": f["path"].stat().st_size,
                "sha256": sha256(f["path"]),
                "rows": f["rows"],
                "kind": f["kind"],
                "description": f["description"],
            }
            for f in files
        ],
    }
    save_json(out / "manifest.json", manifest)
    total = sum(f["bytes"] for f in manifest["files"])
    logger.info("wrote %d files (%.1f MB) and manifest.json to %s", len(files), total / 1e6, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
