"""Strictly typed cleaning actions.

Two layers on purpose:

1. ``DataCleaningAction`` (models.py) is the loose *transport* model. It accepts any ``action_type`` string so
   that a bad action from an LLM reaches the environment instead of being rejected by the web layer.
2. The classes here are the strict *semantic* models. ``parse_action`` validates a transport action against
   them and returns either a typed action or a human-readable error that goes back to the agent.
"""

from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

Strategy = Literal["mean", "median", "mode", "constant"]
TargetType = Literal["int", "float", "str"]

ACTION_TYPES = (
    "fill_missing",
    "drop_duplicates",
    "standardize_categories",
    "cast_type",
    "strip_whitespace",
    "fix_dates",
    "clip_outliers",
    "drop_rows_with_missing",
    "finish",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FillMissing(_Strict):
    action_type: Literal["fill_missing"]
    column_name: str
    strategy: Strategy
    value: Optional[str] = None  # only for strategy="constant"

    @model_validator(mode="after")
    def _constant_needs_value(self) -> "FillMissing":
        if self.strategy == "constant" and self.value is None:
            raise ValueError("strategy 'constant' requires 'value'")
        if self.strategy != "constant" and self.value is not None:
            raise ValueError("'value' is only allowed with strategy 'constant'")
        return self


class DropDuplicates(_Strict):
    action_type: Literal["drop_duplicates"]


class StandardizeCategories(_Strict):
    action_type: Literal["standardize_categories"]
    column_name: str
    mapping: Optional[Dict[str, str]] = None  # None -> automatic (whitespace + case) clean-up


class CastType(_Strict):
    action_type: Literal["cast_type"]
    column_name: str
    target_type: TargetType


class StripWhitespace(_Strict):
    action_type: Literal["strip_whitespace"]
    column_name: Optional[str] = None  # None -> every text column


class FixDates(_Strict):
    action_type: Literal["fix_dates"]
    column_name: str


class ClipOutliers(_Strict):
    action_type: Literal["clip_outliers"]
    column_name: str


class DropRowsWithMissing(_Strict):
    action_type: Literal["drop_rows_with_missing"]
    column_name: Optional[str] = None  # None -> any column


class Finish(_Strict):
    action_type: Literal["finish"]


TypedAction = Annotated[
    Union[
        FillMissing,
        DropDuplicates,
        StandardizeCategories,
        CastType,
        StripWhitespace,
        FixDates,
        ClipOutliers,
        DropRowsWithMissing,
        Finish,
    ],
    Field(discriminator="action_type"),
]
_adapter = TypeAdapter(TypedAction)

# Hints the agent gets with an error so it can fix its next attempt.
USAGE: Dict[str, str] = {
    "fill_missing": '{"action_type":"fill_missing","column_name":C,"strategy":"mean|median|mode|constant","value":V (constant only)}',
    "drop_duplicates": '{"action_type":"drop_duplicates"}',
    "standardize_categories": '{"action_type":"standardize_categories","column_name":C,"mapping":{"NY":"New York"} (optional)}',
    "cast_type": '{"action_type":"cast_type","column_name":C,"target_type":"int|float|str"}',
    "strip_whitespace": '{"action_type":"strip_whitespace","column_name":C (optional)}',
    "fix_dates": '{"action_type":"fix_dates","column_name":C}',
    "clip_outliers": '{"action_type":"clip_outliers","column_name":C}',
    "drop_rows_with_missing": '{"action_type":"drop_rows_with_missing","column_name":C (optional)}',
    "finish": '{"action_type":"finish"}',
}


def parse_action(raw: Dict[str, Any]) -> "tuple[Optional[BaseModel], Optional[str]]":
    """Validate ``raw`` (dict of non-null transport fields). Returns ``(action, None)`` or ``(None, error)``."""
    action_type = raw.get("action_type")
    if action_type not in ACTION_TYPES:
        return None, f"unknown action_type {action_type!r}; valid types: {', '.join(ACTION_TYPES)}"
    try:
        return _adapter.validate_python(raw), None
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'action'}: {e['msg']}" for e in exc.errors()
        )
        return None, f"invalid {action_type} action ({details}). Usage: {USAGE[action_type]}"
