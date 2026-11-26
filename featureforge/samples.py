"""Seeded synthetic *messy* datasets with a known signal, for demos, tests and benchmarks.

``make_dataset(kind, n_rows, seed)`` returns a DataFrame with realistic problems: missing values, placeholders
('N/A', '-', '?'), numbers stored as text ('$1,250.00', '45%'), mixed date formats, case/whitespace variants,
outliers, exact duplicates, a leaky column (``churn`` only, documented) and an ID column.

Kinds: ``churn`` (binary classification), ``housing`` (regression), ``transactions`` (imbalanced classification with
timestamps, a high-cardinality merchant and free text).
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd

CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle", "Boston", "Denver"]
PLACEHOLDERS = ["N/A", "-", "?", "null", ""]
DATE_FMTS = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%Y/%m/%d", "%d %b %Y"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _messy_city(c: str, rng: np.random.Generator) -> str:
    return [c, c.lower(), c.upper(), f" {c}", f"{c}  ", c.replace(" ", "  ")][int(rng.integers(6))]


def _dates(n: int, rng: np.random.Generator, start: str = "2021-01-01", days: int = 1200) -> pd.Series:
    base = pd.Timestamp(start)
    return pd.Series([base + pd.Timedelta(days=int(d)) for d in rng.integers(0, days, n)])


def _mix_formats(dates: pd.Series, rng: np.random.Generator, frac: float) -> list:
    out = []
    for d in dates:
        fmt = DATE_FMTS[int(rng.integers(1, len(DATE_FMTS)))] if rng.random() < frac else DATE_FMTS[0]
        out.append(d.strftime(fmt))
    return out


def _inject_missing(df: pd.DataFrame, cols: Dict[str, float], rng: np.random.Generator) -> pd.DataFrame:
    for c, frac in cols.items():
        idx = rng.choice(len(df), int(frac * len(df)), replace=False)
        df.loc[idx, c] = None
    return df


def _placeholders(df: pd.DataFrame, col: str, frac: float, rng: np.random.Generator) -> None:
    df[col] = df[col].astype(object)
    idx = rng.choice(len(df), int(frac * len(df)), replace=False)
    for i in idx:
        df.at[i, col] = PLACEHOLDERS[int(rng.integers(len(PLACEHOLDERS)))]


def _duplicate(df: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    k = int(frac * len(df))
    if k:
        df = pd.concat([df, df.iloc[rng.choice(len(df), k, replace=False)]], ignore_index=True)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def churn(n_rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows
    age = rng.integers(18, 80, n).astype(float)
    tenure = rng.gamma(2.0, 12.0, n)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.55, 0.33, 0.12])
    spend = np.round(np.where(plan == "basic", 20, np.where(plan == "pro", 60, 220)) * rng.lognormal(0, 0.35, n), 2)
    tickets = rng.poisson(1.2, n)
    city = rng.choice(CITIES, n)
    signup = _dates(n, rng)
    last_login_days = rng.exponential(20, n)
    logit = -1.0 + 0.45 * tickets - 0.04 * tenure + 0.025 * last_login_days - 0.6 * (plan == "enterprise")
    logit += 0.3 * np.isin(city, ["Houston", "Phoenix"])
    y = (rng.random(n) < _sigmoid(logit)).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(100_000, 100_000 + n),
            "signup_date": _mix_formats(signup, rng, 0.35),
            "age": age,
            "city": [_messy_city(c, rng) if rng.random() < 0.3 else c for c in city],
            "plan": [p.upper() if rng.random() < 0.1 else p for p in plan],
            "monthly_spend": [f"${s:,.2f}" if rng.random() < 0.4 else s for s in spend],
            "tenure_months": np.round(tenure, 1),
            "support_tickets": tickets,
            "days_since_login": np.round(last_login_days, 1),
            "autopay": rng.choice(["yes", "no", "Yes", "Y", "N"], n, p=[0.4, 0.35, 0.1, 0.1, 0.05]),
            "feedback": [
                rng.choice(
                    [
                        "great service, very happy",
                        "app keeps crashing when I pay",
                        "too expensive for what it offers",
                        "support was slow to answer my ticket",
                        "no complaints so far",
                    ]
                )
                if rng.random() < 0.7
                else None
                for _ in range(n)
            ],
            "constant_col": "v1",
            # leaky: recorded only after the customer churned (a deliberate trap the profiler must flag)
            "cancellation_reason": np.where(y == 1, rng.choice(["price", "service", "moved"], n), "none"),
            "churned": y,
        }
    )
    outl = rng.choice(n, int(0.01 * n), replace=False)
    df.loc[outl, "age"] = rng.choice([-1, 150, 999], len(outl))
    df = _inject_missing(df, {"age": 0.08, "tenure_months": 0.05, "city": 0.04, "days_since_login": 0.06}, rng)
    _placeholders(df, "age", 0.03, rng)
    return _duplicate(df, 0.03, rng)


def housing(n_rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows
    sqft = rng.lognormal(7.3, 0.35, n)
    beds = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 7)
    age = rng.integers(0, 100, n)
    hood = rng.choice([f"district_{i:02d}" for i in range(40)], n)
    hood_effect = {h: e for h, e in zip(sorted(set(hood)), rng.normal(0, 0.25, 40), strict=False)}
    lot = sqft * rng.uniform(1.2, 4.0, n)
    built = pd.Series(pd.to_datetime("2025-01-01") - pd.to_timedelta(age * 365, unit="D"))
    price = np.exp(
        11.2 + 0.85 * np.log(sqft) - 0.003 * age + np.array([hood_effect[h] for h in hood]) + rng.normal(0, 0.12, n)
    )
    df = pd.DataFrame(
        {
            "listing_id": [f"L{100000 + i}" for i in range(n)],
            "sqft": [f"{s:,.0f}" if rng.random() < 0.3 else round(s) for s in sqft],
            "bedrooms": beds,
            "lot_size": np.round(lot),
            "built_on": _mix_formats(built, rng, 0.4),
            "neighborhood": [h.upper() if rng.random() < 0.15 else h for h in hood],
            "condition": rng.choice(["good", "fair", "excellent", "poor", "Good"], n),
            "has_garage": rng.choice(["true", "false", "TRUE", "False"], n),
            "description": [
                f"{int(b)} bed home with {rng.choice(['garden', 'pool', 'view', 'new kitchen', 'basement'])}"
                for b in beds
            ],
            "price": np.round(price, -2),
        }
    )
    df = _inject_missing(df, {"bedrooms": 0.05, "lot_size": 0.1, "condition": 0.05}, rng)
    _placeholders(df, "lot_size", 0.02, rng)
    return _duplicate(df, 0.02, rng)


def transactions(n_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows
    ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 180 * 24 * 3600, n), unit="s")
    merchants = [f"m_{i:04d}" for i in range(600)]
    merchant = rng.choice(merchants, n, p=np.random.default_rng(seed + 1).dirichlet(np.ones(600) * 0.3))
    risky = set(rng.choice(merchants, 30, replace=False))
    amount = rng.lognormal(3.5, 1.1, n)
    hour = pd.DatetimeIndex(ts).hour.to_numpy()
    country = rng.choice(["US", "us", "GB", "IN", "DE", "BR"], n, p=[0.5, 0.1, 0.15, 0.1, 0.1, 0.05])
    logit = -5.2 + 1.8 * np.isin(merchant, list(risky)) + 0.6 * np.log1p(amount) / 3 + 1.2 * ((hour < 5) | (hour > 22))
    fraud = (rng.random(n) < _sigmoid(logit)).astype(int)
    df = pd.DataFrame(
        {
            "txn_id": [f"T{i:07d}" for i in range(n)],
            "timestamp": pd.Series(ts).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "merchant": merchant,
            "amount": [f"{a:.2f}" if rng.random() < 0.2 else round(a, 2) for a in amount],
            "country": country,
            "channel": rng.choice(["web", "app", "pos", "Web "], n),
            "memo": [
                rng.choice(["online purchase", "store payment", "subscription renewal", "refund request"])
                for _ in range(n)
            ],
            "is_fraud": fraud,
        }
    )
    df = _inject_missing(df, {"channel": 0.03, "country": 0.02}, rng)
    return _duplicate(df, 0.01, rng)


DATASETS: Dict[str, Callable[..., pd.DataFrame]] = {"churn": churn, "housing": housing, "transactions": transactions}
TARGETS = {"churn": "churned", "housing": "price", "transactions": "is_fraud"}


def make_dataset(kind: str = "churn", n_rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    if kind not in DATASETS:
        raise ValueError(f"unknown sample {kind!r}; choose one of {sorted(DATASETS)}")
    return DATASETS[kind](n_rows=n_rows, seed=seed)
