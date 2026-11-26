"""Reward functions for GRPO: score a generated action with the environment itself.

Each dataset row carries ``task_id``, ``seed`` and ``prefix`` (the actions already applied). The reward function
rebuilds that exact state (environments are deterministic), applies the model's action and returns the
environment's own per-step reward (potential-based shaping + penalties, see ``data_cleaning_env_environment.py``).
A reply that is not one valid JSON action gets ``INVALID_REWARD``. A small format bonus rewards clean, bare JSON.
"""

import json
from functools import lru_cache
from typing import Any, List, Sequence

INVALID_REWARD = -1.0
FORMAT_BONUS = 0.1


def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


@lru_cache(maxsize=4096)
def _replay(task_id: str, seed: int, prefix: str):
    """Environment state after the prefix actions (cached; copied before use)."""
    from data_cleaning_env.models import DataCleaningAction
    from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment

    env = DataCleaningEnvironment()
    env.reset(task_id=task_id, seed=seed)
    for payload in json.loads(prefix):
        env.step(DataCleaningAction(**payload))
    return env


def score_action(task_id: str, seed: int, prefix: str, reply: str) -> float:
    import copy

    from agents.llm_agent import validate_reply

    try:
        action = validate_reply(reply)
    except ValueError:
        return INVALID_REWARD
    env = copy.deepcopy(_replay(task_id, int(seed), prefix))
    obs = env.step(action)
    if not obs.last_action_ok:
        return INVALID_REWARD
    return float(obs.reward or 0.0)


def env_reward(
    completions: Sequence[Any], task_id: Sequence[str], seed: Sequence[int], prefix: Sequence[str], **_
) -> List[float]:
    """GRPO reward: the environment's step reward for the generated action."""
    return [score_action(t, s, p, _text(c)) for c, t, s, p in zip(completions, task_id, seed, prefix, strict=True)]


def format_reward(completions: Sequence[Any], **_) -> List[float]:
    """Small bonus when the reply is exactly one JSON object (no prose, no code fences)."""
    out = []
    for c in completions:
        t = _text(c).strip()
        try:
            ok = t.startswith("{") and t.endswith("}") and isinstance(json.loads(t), dict)
        except ValueError:
            ok = False
        out.append(FORMAT_BONUS if ok else 0.0)
    return out
