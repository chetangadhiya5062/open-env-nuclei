"""Run the LLM agent against a running environment server.

Kept at the repository root because the Meta x Scaler OpenEnv hackathon submission requires a root-level
``inference.py``. Stdout uses the [START] / [STEP] / [END] line format; diagnostics go through ``logging``
(stderr).

    $env:HF_TOKEN = "hf_..."          # see .env.example
    python -m data_cleaning_env.server.app      # terminal 1 (port 7860)
    python inference.py                         # terminal 2 (all three tasks)
    python inference.py --task hard-clean --seed 3

Environment variables: HF_TOKEN, API_BASE_URL, MODEL_NAME, ENV_URL (default http://localhost:7860).
"""

import argparse
import json
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

from agents import LLMAgent, RemoteEnv, run_episode
from data_cleaning_env.server.datagen import TASKS

BENCHMARK_NAME = "data_cleaning_env"
SUCCESS_THRESHOLD = 0.8  # grader score at or above which an episode is reported as a success

logger = logging.getLogger("inference")


def run_task(agent: LLMAgent, env: RemoteEnv, task_id: str, seed: int) -> float:
    rewards: List[float] = []
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={agent.model}", flush=True)

    def on_step(step, action, obs):
        rewards.append(float(obs.reward or 0.0))
        error = json.dumps(obs.last_error) if obs.last_error else "null"
        print(
            f"[STEP] step={step} action={json.dumps(action.to_payload(), separators=(',', ':'))} "
            f"reward={rewards[-1]:.2f} done={str(obs.done).lower()} error={error}",
            flush=True,
        )

    result = run_episode(agent, env, task_id, seed, on_step=on_step)
    print(
        f"[END] success={str(result.final_score >= SUCCESS_THRESHOLD).lower()} steps={result.steps} "
        f"score={result.final_score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )
    return result.final_score


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the LLM agent on the data-cleaning environment")
    parser.add_argument("--task", choices=sorted(TASKS), default=None, help="default: run all tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-url", default=os.getenv("ENV_URL", "http://localhost:7860"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    if not LLMAgent.available():
        logger.error("HF_TOKEN is not set. Copy .env.example to .env and fill it in.")
        return 1
    agent = LLMAgent()
    env = RemoteEnv(args.env_url)
    try:
        for task_id in [args.task] if args.task else list(TASKS):
            run_task(agent, env, task_id, args.seed)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
