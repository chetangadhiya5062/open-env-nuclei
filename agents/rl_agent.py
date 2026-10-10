"""Trained RL policy as an agent. Inference is plain numpy, so torch / stable-baselines3 are NOT needed to run it.

The weights come from ``train_rl.py`` (PPO, MLP 2x64 tanh) exported to ``models/ppo_policy.npz``.
The policy acts greedily (argmax over action logits).
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation

from .base import Agent
from .rl_common import N_ACTIONS, N_FEATURES, build_action, featurize

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ppo_policy.npz"


class NumpyPolicy:
    """tanh MLP: obs -> [hidden layers] -> action logits."""

    def __init__(self, weights: dict):
        self.hidden = []
        i = 0
        while f"W{i}" in weights:
            self.hidden.append((weights[f"W{i}"], weights[f"b{i}"]))
            i += 1
        self.out_w, self.out_b = weights["Wa"], weights["ba"]
        assert self.hidden[0][0].shape[1] == N_FEATURES and self.out_w.shape[0] == N_ACTIONS, "model/feature mismatch"

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NumpyPolicy":
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32)
        for w, b in self.hidden:
            h = np.tanh(h @ w.T + b)
        return h @ self.out_w.T + self.out_b

    def act(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(x)))


class RLAgent(Agent):
    name = "rl-ppo"

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.policy = NumpyPolicy.load(self.model_path)
        self._initial_rows: Optional[int] = None

    @staticmethod
    def available(model_path: Optional[Union[str, Path]] = None) -> bool:
        return Path(model_path or DEFAULT_MODEL_PATH).exists()

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self._initial_rows = None

    def act(self, obs: DataCleaningObservation) -> DataCleaningAction:
        if self._initial_rows is None:  # the first observation of an episode is the untouched table
            self._initial_rows = obs.row_count
        index = self.policy.act(featurize(obs, self._initial_rows))
        return build_action(index, obs)
