"""Typed models for the Data Cleaning environment (action, observation, state)."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

TaskId = Literal["easy-clean", "medium-clean", "hard-clean"]


class DataCleaningAction(Action):
    """Transport model for one cleaning action.

    ``action_type`` is intentionally a plain string: unknown or malformed actions must reach the
    environment so they can be answered with an informative error and a small penalty. The strict
    per-action models live in :mod:`data_cleaning_env.actions`.

    action_type  | required fields                       | optional
    -------------|---------------------------------------|----------------------------
    fill_missing | column_name, strategy                 | value (strategy="constant")
    drop_duplicates | -                                  |
    standardize_categories | column_name                 | mapping {"NY": "New York"}
    cast_type    | column_name, target_type (int/float/str) |
    strip_whitespace | -                                 | column_name
    fix_dates    | column_name                           |
    clip_outliers | column_name                          |
    drop_rows_with_missing | -                           | column_name
    finish       | -                                     |
    """

    action_type: str
    column_name: Optional[str] = None
    strategy: Optional[str] = None  # mean | median | mode | constant
    value: Optional[str] = None
    mapping: Optional[Dict[str, str]] = None
    target_type: Optional[str] = None  # int | float | str

    def to_payload(self) -> Dict[str, Any]:
        """Only the fields that are set (what the strict action models expect)."""
        return self.model_dump(exclude_none=True, exclude={"metadata"})


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    expected_type: str
    missing: int
    unique: int
    sample_values: List[str]
    issues: Dict[str, int] = Field(default_factory=dict, description="detected problems -> number of cells")
    allowed_values: Optional[List[str]] = None
    valid_range: Optional[List[float]] = None


class DataCleaningObservation(Observation):
    """What the agent sees after each step. ``reward`` is the reward of the last step only."""

    task_id: str = "easy-clean"
    seed: int = 0
    row_count: int = 0
    duplicate_count: int = 0
    total_missing: int = 0
    columns: List[ColumnProfile] = Field(default_factory=list)
    data_sample: List[Dict[str, Any]] = Field(default_factory=list)

    step_count: int = 0
    steps_remaining: int = 0
    quality_score: float = Field(0.0, description="observable data-quality score in [0, 1] (used for reward shaping)")
    total_reward: float = Field(0.0, description="sum of per-step rewards so far this episode")
    invalid_action_count: int = 0

    last_action: Optional[str] = None
    last_action_ok: bool = True
    last_message: str = ""
    last_error: Optional[str] = None

    # Only set once the episode is over; computed against the hidden ground truth.
    final_score: Optional[float] = Field(None, description="grader score in [0, 1], set when done")
    score_breakdown: Optional[Dict[str, float]] = None


class DataCleaningState(State):
    task_id: str = "easy-clean"
    seed: int = 0
    max_steps: int = 0
    total_reward: float = 0.0
