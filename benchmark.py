"""Benchmark agents on every task over several seeds.

Examples (PowerShell):
    python benchmark.py                                  # random + rule-based + do-nothing, 10 seeds
    python benchmark.py --agents random,rule --seeds 20
    $env:HF_TOKEN = "hf_..."; python benchmark.py --agents llm --seeds 5   # LLM only

Outputs (default folder ``benchmark_results/``): ``episodes.csv`` (one row per episode),
``results.md`` (summary table) and ``scores.png`` (bar chart of mean score +/- std).
Agents that need an API key are skipped, with a warning, when HF_TOKEN is missing.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
from dotenv import load_dotenv

from agents import DoNothingAgent, LLMAgent, LocalEnv, RandomAgent, RemoteEnv, RuleBasedAgent, run_episode
from agents.base import Agent
from data_cleaning_env.server.datagen import TASKS

logger = logging.getLogger("benchmark")

AGENT_FACTORIES: Dict[str, Callable[[], Agent]] = {
    "do-nothing": DoNothingAgent,
    "random": RandomAgent,
    "rule": RuleBasedAgent,
    "llm": LLMAgent,
}
DEFAULT_AGENTS = ["do-nothing", "random", "rule"]


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    grouped = episodes.groupby(["agent", "task_id"], sort=False)
    out = grouped.agg(
        episodes=("seed", "count"),
        score_mean=("final_score", "mean"),
        score_std=("final_score", "std"),
        reward_mean=("total_reward", "mean"),
        steps_mean=("steps", "mean"),
        invalid=("invalid_actions", "sum"),
        total_steps=("steps", "sum"),
        fallbacks=("fallbacks", "sum"),
    ).reset_index()
    out["score_std"] = out["score_std"].fillna(0.0)
    out["invalid_rate"] = out["invalid"] / out["total_steps"].clip(lower=1)
    return out


def to_markdown(summary: pd.DataFrame, n_seeds: int) -> str:
    lines = [
        f"Mean over {n_seeds} seeds per task (score = grader score in [0, 1]; higher is better).",
        "",
        "| Agent | Task | Score (mean ± std) | Mean reward | Mean steps | Invalid-action rate | LLM fallbacks |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.agent} | {r.task_id} | {r.score_mean:.3f} ± {r.score_std:.3f} | {r.reward_mean:.2f} | "
            f"{r.steps_mean:.1f} | {r.invalid_rate:.1%} | {int(r.fallbacks)} |"
        )
    return "\n".join(lines) + "\n"


def plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(summary["task_id"]))
    agents = list(dict.fromkeys(summary["agent"]))
    width = 0.8 / max(1, len(agents))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, agent in enumerate(agents):
        sub = summary[summary["agent"] == agent].set_index("task_id").reindex(tasks)
        xs = [t + i * width for t in range(len(tasks))]
        ax.bar(xs, sub["score_mean"], width, yerr=sub["score_std"], capsize=3, label=agent)
    ax.set_xticks([t + width * (len(agents) - 1) / 2 for t in range(len(tasks))])
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("grader score (mean ± std)")
    ax.set_title("Data-cleaning benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: List[str] = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENTS), help=f"comma list from {sorted(AGENT_FACTORIES)}")
    parser.add_argument("--tasks", default=",".join(TASKS), help="comma list of task ids")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds per task (0..N-1)")
    parser.add_argument("--out", default="benchmark_results")
    parser.add_argument("--env-url", default=None, help="run against a server (WebSocket) instead of in-process")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rows = []
    env = RemoteEnv(args.env_url) if args.env_url else LocalEnv()
    try:
        for agent_key in [a.strip() for a in args.agents.split(",") if a.strip()]:
            if agent_key not in AGENT_FACTORIES:
                logger.error("unknown agent %r (choose from %s)", agent_key, sorted(AGENT_FACTORIES))
                return 2
            if agent_key == "llm" and not LLMAgent.available():
                logger.warning("skipping 'llm': HF_TOKEN is not set")
                continue
            agent = AGENT_FACTORIES[agent_key]()
            for task_id in [t.strip() for t in args.tasks.split(",")]:
                for seed in range(args.seeds):
                    result = run_episode(agent, env, task_id, seed)
                    rows.append(result.as_dict())
                    logger.info(
                        "%s %s seed=%d score=%.3f steps=%d",
                        result.agent,
                        task_id,
                        seed,
                        result.final_score,
                        result.steps,
                    )
    finally:
        env.close()

    if not rows:
        logger.error("no episodes were run")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = pd.DataFrame(rows)
    episodes.to_csv(out / "episodes.csv", index=False)
    summary = summarize(episodes)
    md = to_markdown(summary, args.seeds)
    (out / "results.md").write_text(md, encoding="utf-8")
    plot(summary, out / "scores.png")
    print(md)
    print(f"Wrote {out / 'episodes.csv'}, {out / 'results.md'}, {out / 'scores.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
