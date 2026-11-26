"""Evaluate an exported PPO policy (``ppo_policy.npz``) on held-out test seeds next to the baselines.

    python -m training.evaluate_ppo --model training/runs/ppo/<run>/ppo_policy.npz
    python -m training.evaluate_ppo --model models/ppo_policy.npz --seeds 10 --smoke

Uses the same in-process episode runner and metrics as ``benchmark.py`` (grader score and improvement over
do-nothing, test seeds 0..N-1). Output: ``<model folder>/test_results.md`` and ``test_episodes.csv``.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from training.common import setup

logger = logging.getLogger("training.evaluate_ppo")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 task")
    args = ap.parse_args()
    setup()
    from agents import DoNothingAgent, LocalEnv, RLAgent, RuleBasedAgent, run_episode
    from data_cleaning_env.server.datagen import TASKS

    model = Path(args.model)
    tasks = ["easy-clean"] if args.smoke else list(TASKS)
    seeds = range(1 if args.smoke else args.seeds)
    env = LocalEnv()
    rows = []
    for name, factory in (
        ("do-nothing", DoNothingAgent),
        ("rule-based", RuleBasedAgent),
        ("ppo", lambda: RLAgent(model)),
    ):
        for task in tasks:
            for s in seeds:
                r = run_episode(factory(), env, task, s).as_dict()
                r["agent"] = name
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(model.parent / "test_episodes.csv", index=False)
    table = df.groupby(["agent", "task_id"], sort=False).agg(
        score=("final_score", "mean"),
        score_std=("final_score", "std"),
        improvement=("improvement", "mean"),
        invalid=("invalid_actions", "sum"),
        steps=("steps", "sum"),
    )
    lines = [
        f"# PPO test evaluation: `{model}`",
        "",
        f"Test seeds 0..{len(seeds) - 1}.",
        "",
        "| agent | task | score | improvement | invalid-action rate |",
        "|---|---|---|---|---|",
    ]
    for (agent, task), r in table.iterrows():
        lines.append(
            f"| {agent} | {task} | {r.score:.3f} ± {0 if pd.isna(r.score_std) else r.score_std:.3f} | "
            f"{r.improvement:+.3f} | {100 * r.invalid / max(1, r.steps):.1f}% |"
        )
    (model.parent / "test_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
