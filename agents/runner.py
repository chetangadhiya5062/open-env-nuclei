"""Run one agent for one episode, in-process or against a running server."""

from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Protocol

from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation

from .base import Agent


class EnvAdapter(Protocol):
    def reset(self, task_id: str, seed: int) -> DataCleaningObservation: ...
    def step(self, action: DataCleaningAction) -> DataCleaningObservation: ...
    def close(self) -> None: ...


class LocalEnv:
    """Direct in-process access to the environment class (fast, no network)."""

    def __init__(self) -> None:
        from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment

        self._env = DataCleaningEnvironment()

    def reset(self, task_id: str, seed: int) -> DataCleaningObservation:
        return self._env.reset(task_id=task_id, seed=seed)

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


class RemoteEnv:
    """Talks to a running server over WebSocket (the same path a deployed Space uses)."""

    def __init__(self, base_url: str) -> None:
        from data_cleaning_env import DataCleaningEnv

        self._cm = DataCleaningEnv(base_url=base_url).sync()
        self._env = self._cm.__enter__()

    def reset(self, task_id: str, seed: int) -> DataCleaningObservation:
        return self._env.reset(task_id=task_id, seed=seed).observation

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        return self._env.step(action).observation

    def close(self) -> None:
        self._cm.__exit__(None, None, None)


@dataclass
class EpisodeResult:
    agent: str
    task_id: str
    seed: int
    steps: int
    total_reward: float
    final_score: float
    quality_score: float
    invalid_actions: int
    fallbacks: int

    def as_dict(self) -> dict:
        return asdict(self)


def run_episode(
    agent: Agent,
    env: EnvAdapter,
    task_id: str,
    seed: int,
    on_step: Optional[Callable[[int, DataCleaningAction, DataCleaningObservation], None]] = None,
    max_iterations: int = 200,
) -> EpisodeResult:
    agent.reset(seed)
    obs = env.reset(task_id, seed)
    rewards: List[float] = []
    for _ in range(max_iterations):
        if obs.done:
            break
        action = agent.act(obs)
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
        if on_step:
            on_step(obs.step_count, action, obs)
    return EpisodeResult(
        agent=agent.name,
        task_id=task_id,
        seed=seed,
        steps=obs.step_count,
        total_reward=round(sum(rewards), 6),
        final_score=float(obs.final_score if obs.final_score is not None else 0.0),
        quality_score=obs.quality_score,
        invalid_actions=obs.invalid_action_count,
        fallbacks=agent.fallbacks,
    )
