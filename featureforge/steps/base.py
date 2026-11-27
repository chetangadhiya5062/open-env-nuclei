"""Base class, fit context and registry for pipeline steps.

A step is a small, fitted transformer over a whole DataFrame:

* ``fit(df, ctx)`` learns everything from the training data (statistics, mappings, chosen columns)
* ``transform(df)`` replays exactly that on any data (training, test, production) - it never looks at a target
* ``fit_transform(df, ctx)`` is ``fit`` then ``transform`` unless the step must treat training rows differently
  (out-of-fold target encoding)

``train_only`` steps (deduplication, dropping rows without a target) change the number of rows, so they run during
fitting only and are skipped when new data is transformed.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set, Type

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ..types import ColumnType

logger = logging.getLogger(__name__)

REGISTRY: Dict[str, Type["Step"]] = {}


class StepError(ValueError):
    """A step cannot run with these parameters on this data; the message says why."""


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class FitContext:
    target: Optional[str] = None
    task: str = "none"  # classification | regression | none
    id_columns: List[str] = field(default_factory=list)
    types: Dict[str, ColumnType] = field(default_factory=dict)
    random_state: int = 42
    max_fit_rows: int = 20_000
    classes: Optional[List[Any]] = None
    dropped: Set[str] = field(default_factory=set)  # columns removed by earlier steps (drop_columns, selection)

    @property
    def protected(self) -> Set[str]:
        return set(self.id_columns) | ({self.target} if self.target else set())

    def y(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if self.target and self.target in df.columns:
            return df[self.target]
        return None

    def columns_of(
        self,
        df: pd.DataFrame,
        kinds: Iterable[ColumnType],
        requested: Optional[List[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """Columns to act on: the ``requested`` ones (validated) or every column of the given ``kinds``."""
        exclude = self.protected if exclude is None else exclude
        if requested is not None:
            unknown = [c for c in requested if c not in df.columns and c not in self.dropped]
            if unknown:
                raise StepError(f"columns not found: {unknown}")
            return [c for c in requested if c not in exclude and c in df.columns]
        kinds = set(kinds)
        return [c for c in df.columns if c not in exclude and self.types.get(c) in kinds]


def register(cls: Type["Step"]) -> Type["Step"]:
    if cls.name in REGISTRY:
        raise RuntimeError(f"duplicate step name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


class Step:
    name: ClassVar[str] = "step"
    kind: ClassVar[str] = "clean"  # clean | feature | encode | select
    description: ClassVar[str] = ""
    Params: ClassVar[Type[BaseModel]] = NoParams
    train_only: ClassVar[bool] = False
    needs_target: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        self.params = self.Params(**params)
        self.notes_: List[str] = []
        self.fitted_ = False

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "Step":
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def fit_transform(self, df: pd.DataFrame, ctx: FitContext) -> pd.DataFrame:
        if self.needs_target and ctx.y(df) is None:
            raise StepError(f"step {self.name!r} needs a target column; set 'target' in the recipe")
        self.fit(df, ctx)
        self.fitted_ = True
        return self.transform(df)

    def note(self, text: str) -> None:
        self.notes_.append(text)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params.model_dump()})"


def subsample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    return df.sample(n, random_state=random_state) if len(df) > n else df
