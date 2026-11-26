"""Validation benchmark for the FeatureForge pipelines: does the cleaning + feature engineering help a model?

    python -m training.fe_benchmark --config training/configs/fe_benchmark.yaml
    python -m training.fe_benchmark --set datasets=[churn] folds=3 --run-name quick
    python -m training.fe_benchmark --smoke

For every dataset (``training/datasets/tabular/<name>_train.parquet``), pipeline variant and model:

* K-fold cross-validation where the **whole pipeline is fitted inside each training fold** and only applied to the
  validation fold (no statistic ever sees validation rows: leakage-safe by construction);
* a final fit on the full training file, scored once on the untouched ``<name>_holdout.parquet``;
* optional Optuna tuning of the gradient-boosting model inside the training folds (``tune.enabled``).

Pipeline variants: ``minimal`` (type fixing + median/constant imputation + ordinal codes: what a quick manual
baseline does), ``clean`` / ``standard`` / ``full`` (``featureforge.auto_recipe`` levels).
Metrics: ROC-AUC + log-loss (classification), RMSE + R² (regression). Output: ``results.md``, ``results.csv``.
This is CPU-heavy (no GPU needed); run it on the GPU machine's CPU cores while the GPU trains.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from training.common import (
    DATASETS,
    apply_smoke,
    base_parser,
    environment_info,
    load_config,
    run_dir,
    save_json,
    seed_everything,
    setup,
)

logger = logging.getLogger("training.fe_benchmark")


def minimal_recipe(target: str):
    from featureforge import Recipe, StepSpec

    return Recipe(
        name="minimal",
        target=target,
        steps=[
            StepSpec(step="fix_types"),
            StepSpec(step="impute", params={"add_indicator": False}),
            StepSpec(step="ordinal_encode"),
        ],
    )


def make_recipe(variant: str, df: pd.DataFrame, target: str):
    from featureforge import auto_recipe

    if variant == "minimal":
        return minimal_recipe(target)
    return auto_recipe(df, target=target, level=variant)


def to_xy(df: pd.DataFrame, target: str, ids: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[c for c in [target, *ids] if c in df.columns])
    X = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]].astype(float)
    return X, df[target] if target in df.columns else None


def make_model(kind: str, task: str, params: Dict[str, Any], seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "hgb":
        cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
        return cls(random_state=seed, **params)
    head = LogisticRegression(max_iter=2000) if task == "classification" else Ridge(alpha=1.0)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), head)


def score(task: str, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score

    if task == "classification":
        p = model.predict_proba(X)[:, 1]
        return {"roc_auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}
    pred = model.predict(X)
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))}


def prepare_target(df: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    from featureforge.types import parse_numeric

    df = df[df[target].notna()].reset_index(drop=True)
    if task == "regression":
        df[target] = parse_numeric(df[target])
    else:
        df[target] = df[target].astype(int)
    return df


def fit_eval(
    variant: str,
    model_kind: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    task: str,
    params: Dict[str, Any],
    seed: int,
) -> Dict[str, float]:
    from featureforge import Pipeline

    pipe = Pipeline(make_recipe(variant, train, target))
    tr = pipe.fit_transform(train)
    va = pipe.transform(valid)
    ids = pipe.ctx_.id_columns
    Xtr, ytr = to_xy(tr, target, ids)
    Xva, yva = to_xy(va, target, ids)
    Xva = Xva.reindex(columns=Xtr.columns)
    model = make_model(model_kind, task, params, seed)
    model.fit(Xtr, ytr)
    res = score(task, model, Xva, yva)
    res["n_features"] = Xtr.shape[1]
    return res


def tune_hgb(
    variant: str, train: pd.DataFrame, target: str, task: str, cfg: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    """Optuna over HistGradientBoosting parameters, scored by inner CV on the training data only."""
    import optuna
    from sklearn.model_selection import KFold, StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splitter = (StratifiedKFold if task == "classification" else KFold)(
        cfg["inner_folds"], shuffle=True, random_state=seed
    )
    folds = list(splitter.split(train, train[target] if task == "classification" else None))

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=100),
        }
        vals = []
        for tr_idx, va_idx in folds:
            r = fit_eval(variant, "hgb", train.iloc[tr_idx], train.iloc[va_idx], target, task, params, seed)
            vals.append(r["roc_auc"] if task == "classification" else -r["rmse"])
        return float(np.mean(vals))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=cfg["n_trials"])
    return study.best_params


def main() -> int:
    args = base_parser(__doc__, "fe_benchmark.yaml").parse_args()
    setup()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    else:
        cfg.pop("smoke", None)
    seed_everything(cfg["seed"])
    from sklearn.model_selection import KFold, StratifiedKFold

    from featureforge.io import read_table
    from featureforge.samples import TARGETS

    data_dir = Path(cfg["data_dir"]) if cfg.get("data_dir") else DATASETS
    out = run_dir("fe_benchmark_smoke" if args.smoke else "fe_benchmark", args.run_name)
    rows: List[Dict[str, Any]] = []
    for name in cfg["datasets"]:
        target = TARGETS[name]
        task = cfg["tasks"][name]
        train = prepare_target(read_table(data_dir / "tabular" / f"{name}_train.parquet"), target, task)
        holdout = prepare_target(read_table(data_dir / "tabular" / f"{name}_holdout.parquet"), target, task)
        drop = (cfg.get("exclude_columns") or {}).get(name, [])
        train, holdout = train.drop(columns=drop), holdout.drop(columns=drop)
        if cfg.get("max_rows") and len(train) > cfg["max_rows"]:
            train = train.sample(cfg["max_rows"], random_state=cfg["seed"]).reset_index(drop=True)
        splitter = (StratifiedKFold if task == "classification" else KFold)(
            cfg["folds"], shuffle=True, random_state=cfg["seed"]
        )
        folds = list(splitter.split(train, train[target] if task == "classification" else None))
        for variant in cfg["variants"]:
            for model_kind in cfg["models"]:
                params: Dict[str, Any] = dict(cfg.get("hgb_params", {})) if model_kind == "hgb" else {}
                if model_kind == "hgb" and cfg["tune"]["enabled"]:
                    params = tune_hgb(variant, train, target, task, cfg["tune"], cfg["seed"])
                t0 = time.time()
                cv = [
                    fit_eval(variant, model_kind, train.iloc[a], train.iloc[b], target, task, params, cfg["seed"])
                    for a, b in folds
                ]
                hold = fit_eval(variant, model_kind, train, holdout, target, task, params, cfg["seed"])
                row = {
                    "dataset": name,
                    "task": task,
                    "variant": variant,
                    "model": model_kind,
                    "seconds": round(time.time() - t0, 1),
                    "n_features": hold["n_features"],
                    "params": params,
                }
                for k in cv[0]:
                    if k != "n_features":
                        row[f"cv_{k}"] = float(np.mean([c[k] for c in cv]))
                        row[f"cv_{k}_std"] = float(np.std([c[k] for c in cv]))
                        row[f"holdout_{k}"] = hold[k]
                rows.append(row)
                logger.info(
                    "%s %s %s: %s",
                    name,
                    variant,
                    model_kind,
                    {
                        k: round(v, 4)
                        for k, v in row.items()
                        if k.startswith(("cv_", "holdout_")) and not k.endswith("_std")
                    },
                )
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    lines = [
        "# FeatureForge feature-engineering benchmark",
        "",
        f"`python -m training.fe_benchmark` ({cfg['folds']}-fold CV with the pipeline fitted inside each fold, "
        "then one fit on the full train file scored on the untouched holdout file).",
        "",
    ]
    for name, g in df.groupby("dataset", sort=False):
        task = g["task"].iloc[0]
        m = ["roc_auc", "log_loss"] if task == "classification" else ["rmse", "r2"]
        lines += [
            f"## {name} ({task})",
            "",
            f"| variant | model | features | CV {m[0]} | CV {m[1]} | holdout {m[0]} | holdout {m[1]} |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in g.iterrows():
            lines.append(
                f"| {r.variant} | {r.model} | {r.n_features} | {r[f'cv_{m[0]}']:.4f} ± {r[f'cv_{m[0]}_std']:.4f} | "
                f"{r[f'cv_{m[1]}']:.4f} | {r[f'holdout_{m[0]}']:.4f} | {r[f'holdout_{m[1]}']:.4f} |"
            )
        lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(out / "run.json", {"config": cfg, "environment": environment_info()})
    logger.info("results -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
