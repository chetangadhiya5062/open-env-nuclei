"""Fit / transform engine for recipes, with per-step reports and a saved, reloadable artifact.

``Pipeline(recipe).fit_transform(train_df)`` learns every step on the training data and records a
:class:`StepReport` per step (rows/columns before and after, columns added/removed, time, notes).
``transform(new_df)`` replays the fitted steps on new data (train-only steps such as deduplication are skipped) and
validates the input schema first (missing columns are an error, extra columns a warning), in the spirit of
Data Validation for Machine Learning (Breck et al., 2019).

``save(dir)`` writes ``recipe.yaml``, ``pipeline.joblib`` (fitted state) and ``metadata.json``. Only load artifacts
you created yourself: joblib files are pickles and can run code when loaded.
"""

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import pandas as pd

from .config import get_settings
from .profile import infer_task
from .recipe import Recipe
from .steps import REGISTRY, FitContext, Step
from .types import ColumnType, infer_types

logger = logging.getLogger(__name__)

ARTIFACT_FILE = "pipeline.joblib"
RECIPE_FILE = "recipe.yaml"
META_FILE = "metadata.json"


class PipelineError(ValueError):
    """A recipe could not be fitted or applied; the message names the step and the reason."""


@dataclass
class StepReport:
    index: int
    step: str
    kind: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_recipe(recipe: Union[Recipe, dict, str, Path]) -> Recipe:
    if isinstance(recipe, Recipe):
        return recipe
    if isinstance(recipe, dict):
        return Recipe.model_validate(recipe)
    if isinstance(recipe, Path) or (isinstance(recipe, str) and recipe.strip().endswith((".yaml", ".yml", ".json"))):
        return Recipe.from_file(recipe)
    return Recipe.from_text(str(recipe))


class Pipeline:
    def __init__(self, recipe: Union[Recipe, dict, str, Path]) -> None:
        self.recipe = _as_recipe(recipe)
        errors = self.recipe.validate_steps()
        if errors:
            raise PipelineError("invalid recipe:\n- " + "\n- ".join(errors))
        self.steps: List[Step] = [REGISTRY[s.step](**s.params) for s in self.recipe.steps if s.enabled]
        self.fitted = False
        self.report_: List[StepReport] = []
        self.warnings_: List[str] = []

    # ------------------------------------------------------------------------------------------ fitting
    def _context(self, df: pd.DataFrame) -> FitContext:
        r = self.recipe
        if r.target is not None and r.target not in df.columns:
            raise PipelineError(f"target column {r.target!r} not found; columns: {list(df.columns)[:30]}")
        unknown = [c for c in r.id_columns if c not in df.columns]
        if unknown:
            raise PipelineError(f"id_columns not found: {unknown}")
        bad = [c for c in r.type_overrides if c not in df.columns]
        if bad:
            raise PipelineError(f"type_overrides for unknown columns: {bad}")
        settings = get_settings()
        n = settings.profile_sample_rows
        sample = df.sample(n, random_state=r.random_state) if len(df) > n else df
        try:
            types = infer_types(sample, r.type_overrides)
        except ValueError as exc:
            raise PipelineError(f"invalid type override: {exc}") from exc
        ids = list(r.id_columns) + [c for c, t in types.items() if t == ColumnType.ID and c not in r.id_columns]
        ids = [c for c in ids if c != r.target]
        task = r.task
        if r.target is None:
            task = "none"
        elif task == "auto":
            task = infer_task(df[r.target])
        classes = None
        if task == "classification":
            classes = sorted(df[r.target].dropna().astype(str).str.strip().unique().tolist())
        return FitContext(
            target=r.target,
            task=task,
            id_columns=ids,
            types=dict(types),
            random_state=r.random_state,
            max_fit_rows=settings.max_fit_rows,
            classes=classes,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise PipelineError("the input table has no rows")
        ctx = self._context(df)
        self.input_columns_ = list(df.columns)
        self.input_types_ = {c: t.value for c, t in ctx.types.items()}
        self.report_ = []
        out = df.reset_index(drop=True)
        for i, step in enumerate(self.steps, start=1):
            before_cols = list(out.columns)
            rows_before = len(out)
            t0 = time.perf_counter()
            try:
                out = step.fit_transform(out, ctx)
            except Exception as exc:
                raise PipelineError(f"step {i} ({step.name}) failed: {type(exc).__name__}: {exc}") from exc
            before_set, after_set = set(before_cols), set(out.columns)
            self.report_.append(
                StepReport(
                    index=i,
                    step=step.name,
                    kind=step.kind,
                    rows_before=rows_before,
                    rows_after=len(out),
                    cols_before=len(before_cols),
                    cols_after=out.shape[1],
                    added=[c for c in out.columns if c not in before_set],
                    removed=[c for c in before_cols if c not in after_set],
                    seconds=round(time.perf_counter() - t0, 4),
                    notes=list(step.notes_),
                    params=step.params.model_dump(mode="json"),
                )
            )
            logger.info("step %d %s: %s -> %s", i, step.name, (rows_before, len(before_cols)), out.shape)
        self.ctx_ = ctx
        self.output_columns_ = list(out.columns)
        dropped_early: set = set()
        for s in self.steps:
            if s.name == "drop_columns":
                dropped_early |= set(getattr(s, "drop_", []))
        self.required_columns_ = [c for c in self.input_columns_ if c != ctx.target and c not in dropped_early]
        self.fit_rows_ = len(df)
        self.fitted_at_ = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.fitted = True
        return out

    def fit(self, df: pd.DataFrame) -> "Pipeline":
        self.fit_transform(df)
        return self

    # ---------------------------------------------------------------------------------------- applying
    def validate_input(self, df: pd.DataFrame) -> List[str]:
        """Raise ``PipelineError`` for missing required columns; return warnings for everything else."""
        self._check_fitted()
        missing = [c for c in self.required_columns_ if c not in df.columns]
        if missing:
            raise PipelineError(f"input is missing {len(missing)} column(s) the pipeline was fitted on: {missing[:20]}")
        warnings = []
        extra = [c for c in df.columns if c not in self.input_columns_]
        if extra:
            warnings.append(f"ignoring {len(extra)} column(s) not seen during fitting: {extra[:20]}")
        return warnings

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self.warnings_ = self.validate_input(df)
        out = df[[c for c in df.columns if c in self.input_columns_]].reset_index(drop=True)
        for i, step in enumerate(self.steps, start=1):
            if step.train_only:
                continue
            try:
                out = step.transform(out)
            except Exception as exc:
                raise PipelineError(f"step {i} ({step.name}) failed on new data: {type(exc).__name__}: {exc}") from exc
        target = self.ctx_.target
        cols = [c for c in self.output_columns_ if c != target or target in out.columns]
        absent = [c for c in cols if c not in out.columns]
        if absent:
            self.warnings_.append(f"{len(absent)} output column(s) could not be produced and are empty: {absent[:10]}")
        return out.reindex(columns=cols)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise PipelineError("the pipeline is not fitted yet; call fit or fit_transform first")

    # ---------------------------------------------------------------------------------------- reporting
    def summary(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "recipe": self.recipe.name,
            "target": self.ctx_.target,
            "task": self.ctx_.task,
            "id_columns": self.ctx_.id_columns,
            "fit_rows": self.fit_rows_,
            "input_columns": len(self.input_columns_),
            "output_columns": len(self.output_columns_),
            "fitted_at": self.fitted_at_,
            "steps": [r.as_dict() for r in self.report_],
        }

    def metadata(self) -> Dict[str, Any]:
        import numpy
        import sklearn

        from . import __version__

        return {
            "featureforge_version": __version__,
            "pandas_version": pd.__version__,
            "numpy_version": numpy.__version__,
            "sklearn_version": sklearn.__version__,
            "fitted_at": self.fitted_at_,
            "fit_rows": self.fit_rows_,
            "target": self.ctx_.target,
            "task": self.ctx_.task,
            "classes": self.ctx_.classes,
            "id_columns": self.ctx_.id_columns,
            "input_columns": self.input_columns_,
            "input_types": self.input_types_,
            "required_columns": self.required_columns_,
            "output_columns": self.output_columns_,
            "steps": [r.as_dict() for r in self.report_],
        }

    # ------------------------------------------------------------------------------------ persistence
    def save(self, directory: Union[str, Path]) -> Path:
        self._check_fitted()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.recipe.save(d / RECIPE_FILE)
        joblib.dump(self, d / ARTIFACT_FILE, compress=3)
        (d / META_FILE).write_text(json.dumps(self.metadata(), indent=2, default=str), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "Pipeline":
        """Load a saved pipeline. Only load artifacts from a trusted source (joblib uses pickle)."""
        d = Path(directory)
        path = d / ARTIFACT_FILE if d.is_dir() else d
        if not path.exists():
            raise PipelineError(f"no pipeline artifact at {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise PipelineError(f"{path} does not contain a featureforge Pipeline")
        from . import __version__

        meta = d / META_FILE if d.is_dir() else None
        if meta and meta.exists():
            saved = json.loads(meta.read_text(encoding="utf-8")).get("featureforge_version")
            if saved and saved.split(".")[:2] != __version__.split(".")[:2]:
                logger.warning("pipeline was saved with featureforge %s, running %s", saved, __version__)
        return obj
