"""Data Cleaning environment.

Episode: ``reset(task_id, seed)`` builds a dirty table (and a hidden ground truth). The agent sends one
cleaning action per step and ends the episode with ``finish`` (or runs out of step budget).

Reward (per step)
-----------------
Let ``Q`` be the *observable* data-quality score in [0, 1] (see ``quality.py``; no ground truth involved).

    r_t =  SHAPING_SCALE * (Q_t - Q_{t-1})        progress, potential-based shaping
         - STEP_COST                              every step
         - NOOP_PENALTY      if the action changed nothing
         - INVALID_PENALTY   if the action was invalid (it is not applied; an error is returned)
         - LOSS_PENALTY * k  where k = ground-truth rows (by customer_id) removed by this step
         + TERMINAL          only on ``finish``: +TERMINAL_BONUS if Q == 1, else -PREMATURE_PENALTY * (1 - Q)

Why this cannot be farmed: the shaping terms telescope (the sum over an episode is
``SHAPING_SCALE * (Q_T - Q_0)``), so cycling through states or repeating an action earns nothing, only
costs ``STEP_COST``. The terminal term is paid at most once because ``finish`` ends the episode. The
reward is *not* the grader: the grader (``grader.py``) compares against the hidden ground truth and is
revealed only in the final observation.
"""

import logging
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment

from .. import actions as A
from ..models import ColumnProfile, DataCleaningAction, DataCleaningObservation, DataCleaningState
from ..schema import COLUMNS, ID_COLUMN, SCHEMA
from . import operations
from .datagen import generate, get_task
from .grader import grade, improvement
from .quality import detect_issues, duplicate_count, issue_cells, quality_score

logger = logging.getLogger(__name__)

SHAPING_SCALE = 10.0
STEP_COST = 0.02
NOOP_PENALTY = 0.05
INVALID_PENALTY = 0.10
LOSS_PENALTY = 0.30
TERMINAL_BONUS = 1.0
PREMATURE_PENALTY = 2.0
DEFAULT_TASK = "easy-clean"
DEFAULT_SEED = 42
SAMPLE_ROWS = 5
MAX_BAD_VALUES = 10


def _py(value: Any) -> Any:
    """numpy/pandas scalar -> plain JSON-friendly Python value."""
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _truth_ids(df: pd.DataFrame) -> set:
    ids = pd.to_numeric(df[ID_COLUMN], errors="coerce").dropna()
    return set(ids.astype("int64").tolist())


class DataCleaningEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._start_episode(DEFAULT_TASK, DEFAULT_SEED)

    # ------------------------------------------------------------------ lifecycle
    def _start_episode(self, task_id: str, seed: int, episode_id: Optional[str] = None) -> None:
        cfg = get_task(task_id)
        self.cfg = cfg
        self.df, self.truth = generate(task_id, seed)
        self.initial_cells = len(self.df) * len(COLUMNS)
        self.do_nothing_score = grade(self.df, self.truth)["score"]  # score if the agent finishes immediately
        self._refresh_quality()
        self.present_ids = _truth_ids(self.df)
        self.invalid_actions = 0
        self.done = False
        self.final: Optional[Dict[str, float]] = None
        self.last_action: Optional[str] = None
        self.last_ok = True
        self.last_message = "episode started"
        self.last_error: Optional[str] = None
        self._state = DataCleaningState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            seed=seed,
            max_steps=cfg.max_steps,
            total_reward=0.0,
        )

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: str = DEFAULT_TASK, **kwargs: Any
    ) -> DataCleaningObservation:
        self._start_episode(task_id, DEFAULT_SEED if seed is None else int(seed), episode_id)
        return self._observe(reward=None)

    def step(
        self, action: DataCleaningAction, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> DataCleaningObservation:
        if self.done:
            self.last_ok, self.last_error = False, "episode is over; call reset()"
            self.last_message = ""
            return self._observe(reward=0.0)

        self._state.step_count += 1
        self.last_action = action.action_type
        typed, error = A.parse_action(action.to_payload())

        reward = -STEP_COST
        finished = False
        prev_df, prev_q, prev_ids = self.df, self.quality, self.present_ids

        if error is None and isinstance(typed, A.Finish):
            finished = True
            self.last_ok, self.last_error, self.last_message = True, None, "episode finished by the agent"
        elif error is None:
            try:
                new_df, message = operations.apply_action(prev_df, typed)
                self.df = new_df
                self.last_ok, self.last_error, self.last_message = True, None, message
            except operations.ActionError as exc:
                error = str(exc)
        if error is not None:
            self.invalid_actions += 1
            self.last_ok, self.last_error, self.last_message = False, error, ""
            reward -= INVALID_PENALTY
        elif not finished and self.df.equals(prev_df):
            reward -= NOOP_PENALTY

        # potential-based shaping + data-destruction penalty
        self._refresh_quality()
        self.present_ids = _truth_ids(self.df)
        reward += SHAPING_SCALE * (self.quality - prev_q)
        reward -= LOSS_PENALTY * len(prev_ids - self.present_ids)

        if finished:
            reward += TERMINAL_BONUS if self.quality >= 1.0 - 1e-9 else -PREMATURE_PENALTY * (1.0 - self.quality)
        truncated = not finished and self._state.step_count >= self.cfg.max_steps
        if finished or truncated:
            self.done = True
            self.final = grade(self.df, self.truth)
            self.final["do_nothing_score"] = self.do_nothing_score
            self.final["improvement"] = round(improvement(self.final["score"], self.do_nothing_score), 6)
            if truncated:
                self.last_message += " (step budget exhausted, episode ended)"
            logger.info(
                "episode over task=%s seed=%s score=%s", self.cfg.task_id, self._state.seed, self.final["score"]
            )

        self._state.total_reward = round(self._state.total_reward + reward, 6)
        return self._observe(reward=round(reward, 6))

    @property
    def state(self) -> DataCleaningState:
        return self._state

    # ------------------------------------------------------------------ observation
    def _refresh_quality(self) -> None:
        """Recompute issues/duplicates/quality once per state change (reused by the reward and the observation)."""
        self._issues = detect_issues(self.df)
        self._dups = duplicate_count(self.df)
        self.quality = quality_score(self.df, self.initial_cells, self._issues, self._dups)

    @staticmethod
    def _bad_values(col: pd.Series, spec) -> list:
        if spec.allowed is None:
            return []
        counts = col.dropna().astype(str).value_counts()
        return [v for v in counts.index if v not in spec.allowed][:MAX_BAD_VALUES]

    def _profile(self) -> list:
        profiles = []
        for spec in SCHEMA:
            col = self.df[spec.name]
            profiles.append(
                ColumnProfile(
                    name=spec.name,
                    dtype=str(col.dtype),
                    expected_type=spec.kind,
                    missing=int(col.isna().sum()),
                    unique=int(col.nunique(dropna=True)),
                    sample_values=[str(v) for v in col.dropna().drop_duplicates().head(3).tolist()],
                    issues=self._issues[spec.name],
                    allowed_values=list(spec.allowed) if spec.allowed else None,
                    bad_values=self._bad_values(col, spec),
                    valid_range=[float(x) for x in spec.valid_range] if spec.valid_range else None,
                )
            )
        return profiles

    def _observe(self, reward: Optional[float]) -> DataCleaningObservation:
        head = self.df.head(SAMPLE_ROWS)
        sample = [{k: _py(v) for k, v in row.items()} for row in head.to_dict(orient="records")]
        return DataCleaningObservation(
            task_id=self.cfg.task_id,
            seed=self._state.seed,
            row_count=len(self.df),
            duplicate_count=self._dups,
            total_missing=int(self.df.isna().sum().sum()),
            columns=self._profile(),
            data_sample=sample,
            step_count=self._state.step_count,
            steps_remaining=max(0, self.cfg.max_steps - self._state.step_count),
            quality_score=round(self.quality, 6),
            total_reward=self._state.total_reward,
            invalid_action_count=self.invalid_actions,
            last_action=self.last_action,
            last_action_ok=self.last_ok,
            last_message=self.last_message,
            last_error=self.last_error,
            final_score=self.final["score"] if self.final else None,
            score_breakdown=self.final if self.final else None,
            reward=reward,
            done=self.done,
        )

    # kept for tests / debugging: how many raw issue cells remain
    def remaining_issue_cells(self) -> int:
        return issue_cells(self.df)
