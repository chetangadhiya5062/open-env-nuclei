"""Seeded synthetic data generator.

``generate(task_id, seed)`` returns ``(dirty, truth)``:

* ``truth``  - the clean ground-truth table (never shown to the agent; used by the grader and
  to detect destroyed rows).
* ``dirty``  - the same table with controlled noise injected.

Everything is driven by one ``numpy`` Generator seeded from ``(seed, task)``, so the same
``(task_id, seed)`` always yields identical data.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..schema import CITIES, COLUMNS, ISO_DATE_FORMAT, PLANS

FIRST_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Ethan",
    "Fiona",
    "George",
    "Hannah",
    "Ivan",
    "Julia",
    "Kiran",
    "Laura",
    "Mohan",
    "Nina",
    "Omar",
    "Priya",
    "Quinn",
    "Rahul",
    "Sara",
    "Tariq",
]
LAST_NAMES = [
    "Smith",
    "Patel",
    "Garcia",
    "Chen",
    "Johnson",
    "Khan",
    "Lopez",
    "Brown",
    "Shah",
    "Miller",
    "Davis",
    "Wilson",
    "Kumar",
    "Taylor",
    "Moore",
]

# Spellings that need an explicit mapping (cannot be fixed by case/whitespace alone).
CITY_ALIASES = {
    "New York": ["NY", "N.Y.", "NYC"],
    "Los Angeles": ["LA", "L.A."],
    "Chicago": ["Chi"],
}
PLACEHOLDER_TOKENS = ["N/A", "n/a", "null", "-", "?", ""]
DATE_FORMATS = [ISO_DATE_FORMAT, "%d %b %Y", "%b %d, %Y", "%Y/%m/%d"]


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    index: int
    difficulty: str
    n_rows: int
    max_steps: int
    description: str
    missing: Dict[str, float] = field(default_factory=dict)  # column -> fraction of cells set to null
    dup_rate: float = 0.0
    city_noise: float = 0.0  # fraction of city cells with case/space/alias variants
    plan_noise: float = 0.0  # fraction of plan cells with case/space variants
    name_whitespace: float = 0.0
    str_numbers: float = 0.0  # numeric cells stored as strings ("25")
    placeholders: float = 0.0  # numeric cells replaced by "N/A"-style tokens
    outliers: float = 0.0  # numeric cells replaced by impossible values
    mixed_dates: float = 0.0  # fraction of dates written in a non-ISO format


TASKS: Dict[str, TaskConfig] = {
    t.task_id: t
    for t in [
        TaskConfig(
            "easy-clean",
            0,
            "easy",
            n_rows=40,
            max_steps=15,
            description="Dataset with only missing values in one column",
            missing={"age": 0.2},
        ),
        TaskConfig(
            "medium-clean",
            1,
            "medium",
            n_rows=60,
            max_steps=25,
            description="Dataset with missing values and duplicate rows",
            missing={"age": 0.15, "monthly_spend": 0.12, "city": 0.10},
            dup_rate=0.12,
        ),
        TaskConfig(
            "hard-clean",
            2,
            "hard",
            n_rows=100,
            max_steps=45,
            description=(
                "Dataset with missing values, duplicates, inconsistent categories, "
                "wrong types, whitespace issues, outliers and mixed date formats"
            ),
            missing={"age": 0.10, "monthly_spend": 0.08, "city": 0.08, "plan": 0.08},
            dup_rate=0.10,
            city_noise=0.45,
            plan_noise=0.30,
            name_whitespace=0.25,
            str_numbers=0.15,
            placeholders=0.06,
            outliers=0.05,
            mixed_dates=0.5,
        ),
    ]
}


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        raise ValueError(f"unknown task_id {task_id!r}; choose one of {sorted(TASKS)}")
    return TASKS[task_id]


def make_clean(n: int, rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(1001, 1001 + n)
    plans = rng.choice(PLANS, size=n, p=[0.5, 0.35, 0.15])
    spend_lo = {"basic": 10, "pro": 40, "enterprise": 150}
    spend_hi = {"basic": 30, "pro": 90, "enterprise": 400}
    spend = [round(float(rng.uniform(spend_lo[p], spend_hi[p])), 2) for p in plans]
    base = date(2021, 1, 1)
    dates = [(base + timedelta(days=int(d))).strftime(ISO_DATE_FORMAT) for d in rng.integers(0, 1000, size=n)]
    names = [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n)]
    return pd.DataFrame(
        {
            "customer_id": ids.astype("int64"),
            "name": names,
            "age": rng.integers(18, 81, size=n).astype("int64"),
            "city": [str(c) for c in rng.choice(CITIES, size=n)],
            "signup_date": dates,
            "plan": [str(p) for p in plans],
            "monthly_spend": spend,
        }
    )[COLUMNS]


def _pick(rng: np.random.Generator, pool: np.ndarray, frac: float, n: int) -> np.ndarray:
    """Pick ``round(frac*n)`` indices from ``pool`` (without replacement)."""
    k = min(len(pool), int(round(frac * n)))
    return rng.choice(pool, size=k, replace=False) if k > 0 else np.array([], dtype=int)


def _city_variant(city: str, rng: np.random.Generator) -> str:
    options = [city.lower(), city.upper(), f"{city} ", f" {city}"] + CITY_ALIASES.get(city, [])
    return options[int(rng.integers(len(options)))]


def _plan_variant(plan: str, rng: np.random.Generator) -> str:
    options = [plan.upper(), plan.capitalize(), f"{plan} ", f" {plan}"]
    return options[int(rng.integers(len(options)))]


def inject_noise(truth: pd.DataFrame, cfg: TaskConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = len(truth)
    cols = {c: truth[c].astype(object).tolist() for c in truth.columns}

    # numeric value-level noise: each cell gets at most one of {missing, str, placeholder, outlier}
    for col in ("age", "monthly_spend"):
        free = np.arange(n)
        for kind, frac in (
            ("missing", cfg.missing.get(col, 0.0)),
            ("str", cfg.str_numbers),
            ("placeholder", cfg.placeholders),
            ("outlier", cfg.outliers),
        ):
            idx = _pick(rng, free, frac, n)
            free = np.setdiff1d(free, idx)
            for i in idx:
                if kind == "missing":
                    cols[col][i] = None
                elif kind == "str":
                    cols[col][i] = str(cols[col][i])
                elif kind == "placeholder":
                    cols[col][i] = PLACEHOLDER_TOKENS[int(rng.integers(len(PLACEHOLDER_TOKENS)))]
                elif col == "age":
                    cols[col][i] = int(rng.choice([-5, 150, 999]))
                else:
                    cols[col][i] = round(float(rng.uniform(5000, 20000)), 2)

    # categorical noise: nulls first, then spelling variants on the remaining cells
    for col in ("city", "plan"):
        for i in _pick(rng, np.arange(n), cfg.missing.get(col, 0.0), n):
            cols[col][i] = None
    for col, frac, variant in (("city", cfg.city_noise, _city_variant), ("plan", cfg.plan_noise, _plan_variant)):
        present = np.array([i for i in range(n) if cols[col][i] is not None], dtype=int)
        for i in _pick(rng, present, frac, n):
            cols[col][i] = variant(cols[col][i], rng)

    # whitespace in names, mixed date formats
    for i in _pick(rng, np.arange(n), cfg.name_whitespace, n):
        first, last = cols["name"][i].split(" ", 1)
        cols["name"][i] = [f" {first} {last}", f"{first}  {last} ", f"{first} {last} "][int(rng.integers(3))]
    for i in _pick(rng, np.arange(n), cfg.mixed_dates, n):
        fmt = DATE_FORMATS[1 + int(rng.integers(3))]
        cols["signup_date"][i] = date.fromisoformat(cols["signup_date"][i]).strftime(fmt)

    dirty = pd.DataFrame(cols)[COLUMNS]

    # exact duplicate rows, then shuffle
    k = int(round(cfg.dup_rate * n))
    if k:
        dupes = dirty.iloc[rng.choice(n, size=k, replace=False)]
        dirty = pd.concat([dirty, dupes], ignore_index=True)
    return dirty.iloc[rng.permutation(len(dirty))].reset_index(drop=True)


def generate(task_id: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(dirty, truth)`` for a task and seed (deterministic)."""
    cfg = get_task(task_id)
    rng = np.random.default_rng([int(seed), cfg.index])
    truth = make_clean(cfg.n_rows, rng)
    return inject_noise(truth, cfg, rng), truth
