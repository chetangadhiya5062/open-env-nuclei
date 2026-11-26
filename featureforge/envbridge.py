"""Bridge to the OpenEnv data-cleaning environment: score FeatureForge's cleaning on the environment's tasks.

The environment (``data_cleaning_env``) generates a dirty table with a hidden ground truth and grades a cleaned table
with ``grader.grade``. Here the library cleans the dirty table *in one shot* with a generic recipe, using only what an
agent can also see in the observation (expected column types and valid ranges), and the environment's own grader
scores the result. This lets the product be compared with the do-nothing, rule-based, PPO and LLM agents on the same
held-out seeds (``scripts/eval_library_on_env.py``).

What the recipe does *not* get: the allowed category vocabulary, so aliases such as ``NY -> New York`` stay unfixed
(only case/spacing variants are merged). That is a deliberate, stated handicap.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .pipeline import Pipeline
from .recipe import Recipe, StepSpec


def env_recipe() -> Recipe:
    from data_cleaning_env.schema import SCHEMA

    bounds = {c.name: list(c.valid_range) for c in SCHEMA if c.valid_range}
    overrides = {
        c.name: {
            "int": "numeric",
            "float": "numeric",
            "cat": "categorical",
            "text": "categorical",
            "date": "datetime",
            "id": "id",
        }[c.kind]
        for c in SCHEMA
    }
    return Recipe(
        name="openenv-clean",
        id_columns=[c.name for c in SCHEMA if c.kind == "id"],
        type_overrides=overrides,
        steps=[
            StepSpec(step="normalize_strings"),
            StepSpec(step="fix_types"),
            StepSpec(step="deduplicate"),
            StepSpec(step="outliers", params={"bounds": bounds, "action": "clip"}),
            StepSpec(
                step="impute",
                params={"numeric_strategy": "median", "categorical_strategy": "most_frequent", "add_indicator": False},
            ),
        ],
    )


def to_env_format(df: pd.DataFrame) -> pd.DataFrame:
    """Format a cleaned table the way the environment's schema expects (ints, rounded floats, ISO date strings)."""
    from data_cleaning_env.schema import COLUMNS, SCHEMA

    out = df.copy()
    for spec in SCHEMA:
        if spec.name not in out.columns:
            continue
        col = out[spec.name]
        if spec.kind in ("int", "id"):
            num = pd.to_numeric(col, errors="coerce")
            out[spec.name] = num.round().astype("Int64").astype(object).where(num.notna(), None)
            if num.notna().all():
                out[spec.name] = num.round().astype("int64")
        elif spec.kind == "float":
            out[spec.name] = pd.to_numeric(col, errors="coerce").round(2)
        elif spec.kind == "date":
            dt = pd.to_datetime(col, errors="coerce")
            out[spec.name] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), None)
        else:
            out[spec.name] = col.astype(object).where(col.notna(), None)
    return out[[c for c in COLUMNS if c in out.columns]]


def clean_and_grade(task_id: str, seed: int) -> Dict[str, float]:
    from data_cleaning_env.server.datagen import generate
    from data_cleaning_env.server.grader import grade, improvement

    dirty, truth = generate(task_id, seed)
    cleaned = to_env_format(Pipeline(env_recipe()).fit_transform(dirty))
    res = grade(cleaned, truth)
    base = grade(dirty, truth)["score"]
    res["do_nothing_score"] = base
    res["improvement"] = round(float(improvement(res["score"], base)), 6)
    res["rows_out"] = int(len(cleaned))
    res["nan_cells"] = int(cleaned.isna().sum().sum())
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}
