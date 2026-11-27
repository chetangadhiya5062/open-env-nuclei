"""Feature selection (variance, correlation, mutual information) and OpenFE-style candidate generation."""

from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..types import ColumnType, parse_numeric
from .base import FitContext, Step, StepError, register, subsample
from .features import apply_op

EPS = 1e-12


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _numeric_cols(df: pd.DataFrame, ctx: FitContext) -> List[str]:
    return [
        c
        for c in df.columns
        if c not in ctx.protected and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def _target_codes(y: pd.Series, ctx: FitContext) -> Tuple[np.ndarray, bool]:
    """(values, is_discrete) with NaN rows kept as NaN for the caller to mask."""
    if ctx.task == "regression":
        return parse_numeric(y).to_numpy(dtype=float), False
    codes = y.astype(str).where(y.notna())
    cats = {k: i for i, k in enumerate(sorted(codes.dropna().unique()))}
    return codes.map(cats).to_numpy(dtype=float), True


def mi_scores(X: pd.DataFrame, y: pd.Series, ctx: FitContext) -> Dict[str, float]:
    """Mutual information of each numeric column with the target (Kraskov / Ross estimators via scikit-learn)."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    yv, discrete = _target_codes(y, ctx)
    ok = ~np.isnan(yv)
    if ok.sum() < 10 or X.shape[1] == 0:
        return {c: 0.0 for c in X.columns}
    Xn = X.loc[ok].astype(float)
    Xn = Xn.fillna(Xn.median()).fillna(0.0)
    fn = mutual_info_classif if discrete else mutual_info_regression
    scores = fn(Xn.to_numpy(), yv[ok], random_state=ctx.random_state)
    return dict(zip(X.columns, (float(s) for s in scores), strict=True))


# ---------------------------------------------------------------------------------------- variance filter
class VarianceParams(_P):
    threshold: float = Field(0.0, ge=0)


@register
class VarianceFilter(Step):
    name = "variance_filter"
    kind = "select"
    description = "Drop constant and near-constant columns."
    Params = VarianceParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "VarianceFilter":
        drop = []
        for c in df.columns:
            if c in ctx.protected:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                v = df[c].astype(float).var()
                if pd.isna(v) or v <= self.params.threshold:
                    drop.append(c)
            elif df[c].nunique(dropna=False) <= 1:
                drop.append(c)
        self.drop_ = drop
        for c in drop:
            ctx.types.pop(c, None)
        self.note(f"dropped {len(drop)} low-variance column(s)" + (f": {drop[:10]}" if drop else ""))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in self.drop_ if c in df.columns])


# ------------------------------------------------------------------------------------- correlation filter
class CorrelationParams(_P):
    threshold: float = Field(0.95, gt=0, le=1)


@register
class CorrelationFilter(Step):
    name = "correlation_filter"
    kind = "select"
    description = (
        "For each pair with |Pearson r| above the threshold, drop the column less informative about the target."
    )
    Params = CorrelationParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "CorrelationFilter":
        sdf = subsample(df, ctx.max_fit_rows, ctx.random_state)
        cols = [c for c in _numeric_cols(sdf, ctx) if sdf[c].std() > 0]
        y = ctx.y(sdf)
        score = mi_scores(sdf[cols], y, ctx) if y is not None and ctx.task != "none" else {}
        corr = sdf[cols].astype(float).corr().abs()
        pairs = [(corr.loc[a, b], a, b) for a, b in combinations(cols, 2) if corr.loc[a, b] >= self.params.threshold]
        pairs.sort(key=lambda t: -t[0])
        drop: List[str] = []
        for _r, a, b in pairs:
            if a in drop or b in drop:
                continue
            loser = b if score.get(a, 0.0) >= score.get(b, 0.0) else a
            drop.append(loser)
        self.drop_ = drop
        for c in drop:
            ctx.types.pop(c, None)
        self.note(f"dropped {len(drop)} highly correlated column(s)" + (f": {drop[:10]}" if drop else ""))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in self.drop_ if c in df.columns])


# ------------------------------------------------------------------------------------ mutual information
class MIParams(_P):
    k: Optional[int] = Field(None, ge=1, description="keep the k best numeric features")
    fraction: Optional[float] = Field(None, gt=0, le=1, description="keep this share of numeric features")
    min_score: float = Field(0.0, ge=0, description="drop features with MI at or below this")


@register
class MutualInfoSelect(Step):
    name = "mutual_info_select"
    kind = "select"
    description = "Keep the numeric features with the highest mutual information with the target."
    Params = MIParams
    needs_target = True

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "MutualInfoSelect":
        y = ctx.y(df)
        if y is None or ctx.task == "none":
            raise StepError("mutual_info_select needs a target and a classification/regression task")
        sdf = subsample(df, ctx.max_fit_rows, ctx.random_state)
        cols = _numeric_cols(sdf, ctx)
        scores = mi_scores(sdf[cols], sdf[ctx.target], ctx)
        ranked = sorted(cols, key=lambda c: -scores[c])
        keep = [c for c in ranked if scores[c] > self.params.min_score] if self.params.min_score > 0 else ranked
        if self.params.k is not None:
            keep = keep[: self.params.k]
        if self.params.fraction is not None:
            keep = keep[: max(1, int(round(self.params.fraction * len(cols))))]
        self.scores_ = {c: round(scores[c], 6) for c in ranked}
        self.drop_ = [c for c in cols if c not in keep]
        non_numeric = [c for c in df.columns if c not in cols and c not in ctx.protected]
        for c in self.drop_:
            ctx.types.pop(c, None)
        self.note(f"kept {len(keep)} of {len(cols)} numeric feature(s)")
        if non_numeric:
            self.note(f"non-numeric columns left untouched: {non_numeric[:10]}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in self.drop_ if c in df.columns])


# ------------------------------------------------------------------------------- OpenFE-style generation
class GenerationParams(_P):
    max_base_columns: int = Field(8, ge=2, le=40)
    ops: List[Literal["product", "ratio", "difference", "sum"]] = Field(
        default_factory=lambda: ["product", "ratio", "difference", "sum"]
    )
    group_by: bool = True
    max_group_cardinality: int = Field(50, ge=2)
    stage1_keep: int = Field(40, ge=1, description="candidates kept by the coarse MI screen")
    max_new: int = Field(10, ge=1, description="at most this many features are added")
    min_gain: float = Field(0.01, ge=0, description="minimum relative residual-MSE reduction to keep a candidate")
    base_model: Literal["gbm", "linear"] = Field(
        "gbm", description="model whose residuals candidates must explain; use 'linear' for linear downstream models"
    )
    n_folds: int = Field(3, ge=2, le=10)


Spec = Tuple  # ("op", op, a, b) | ("group_mean", cat, num)


@register
class FeatureGeneration(Step):
    name = "feature_generation"
    kind = "select"
    description = (
        "OpenFE-style search (Zhang et al. 2023): generate pairwise and group-by candidates, screen them by mutual "
        "information, then keep those that reduce the out-of-fold residual error of a base gradient-boosting model."
    )
    Params = GenerationParams
    needs_target = True

    def _candidates(self, df: pd.DataFrame, ctx: FitContext, base: List[str]) -> Dict[str, Spec]:
        cands: Dict[str, Spec] = {}
        sym = {"ratio": "div", "product": "mul", "difference": "sub", "sum": "add"}
        for a, b in combinations(base, 2):
            for op in self.params.ops:
                cands[f"gen__{a}__{sym[op]}__{b}"] = ("op", op, a, b)
                if op in ("ratio", "difference"):
                    cands[f"gen__{b}__{sym[op]}__{a}"] = ("op", op, b, a)
        if self.params.group_by:
            cats = [
                c
                for c in df.columns
                if c not in ctx.protected
                and ctx.types.get(c) == ColumnType.CATEGORICAL
                and 2 <= df[c].nunique() <= self.params.max_group_cardinality
            ]
            for g in cats:
                for v in base[:5]:
                    cands[f"gen__{v}__mean_by__{g}"] = ("group_mean", g, v)
        return cands

    def _compute(self, df: pd.DataFrame, spec: Spec, tables: Optional[dict] = None) -> pd.Series:
        if spec[0] == "op":
            _, op, a, b = spec
            return apply_op(op, df[a].astype(float), df[b].astype(float))
        _, g, v = spec
        key = df[g].astype(str).where(df[g].notna(), "__nan__")
        if tables is not None:
            table, glob = tables
            return key.map(table).astype(float).fillna(glob)
        return df[v].astype(float).groupby(key).transform("mean")

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "FeatureGeneration":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict
        from sklearn.tree import DecisionTreeRegressor

        y = ctx.y(df)
        if y is None or ctx.task == "none":
            raise StepError("feature_generation needs a target and a classification/regression task")
        sdf = subsample(df[df[ctx.target].notna()], ctx.max_fit_rows, ctx.random_state)
        numeric = _numeric_cols(sdf, ctx)
        if not numeric:
            self.specs_: Dict[str, Spec] = {}
            self.note("no numeric features to build from")
            return self
        mi = mi_scores(sdf[numeric], sdf[ctx.target], ctx)
        base = [c for c in sorted(numeric, key=lambda c: -mi[c]) if sdf[c].nunique() > 2][
            : self.params.max_base_columns
        ]
        cands = self._candidates(sdf, ctx, base)
        cand_df = pd.DataFrame({n: self._compute(sdf, s) for n, s in cands.items()}, index=sdf.index)
        cand_df = cand_df.loc[:, cand_df.std() > 0]
        # stage 1: coarse screen by mutual information
        s1 = mi_scores(cand_df, sdf[ctx.target], ctx)
        stage1 = sorted(s1, key=lambda n: -s1[n])[: self.params.stage1_keep]

        # stage 2: feature boosting on residuals of a base model
        X = sdf[numeric].astype(float)
        yv, discrete = _target_codes(sdf[ctx.target], ctx)
        if self.params.base_model == "linear":
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            X = X.fillna(X.median()).fillna(0.0)
        if discrete:
            model = (
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                if self.params.base_model == "linear"
                else HistGradientBoostingClassifier(max_iter=100, random_state=ctx.random_state)
            )
            n_classes = len(np.unique(yv))
            min_count = int(np.bincount(yv.astype(int)).min())
            folds = max(2, min(self.params.n_folds, min_count))
            proba = cross_val_predict(model, X, yv.astype(int), cv=folds, method="predict_proba")
            onehot = np.eye(n_classes)[yv.astype(int)]
            resid = onehot - proba
        else:
            model = (
                make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                if self.params.base_model == "linear"
                else HistGradientBoostingRegressor(max_iter=100, random_state=ctx.random_state)
            )
            pred = cross_val_predict(model, X, yv, cv=self.params.n_folds)
            resid = (yv - pred).reshape(-1, 1)
        rng = np.random.default_rng(ctx.random_state)
        half = rng.permutation(len(sdf))
        tr, te = half[: len(half) // 2], half[len(half) // 2 :]
        base_mse = float(np.mean(resid[te] ** 2)) + EPS
        gains = {}
        for n in stage1:
            x = cand_df[n].to_numpy().reshape(-1, 1)
            x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0)
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=ctx.random_state)
            tree.fit(x[tr], resid[tr])
            after = np.mean((resid[te] - tree.predict(x[te]).reshape(resid[te].shape)) ** 2)
            gains[n] = float((base_mse - after) / base_mse)
        chosen = [n for n in sorted(gains, key=lambda n: -gains[n]) if gains[n] > self.params.min_gain]
        chosen = chosen[: self.params.max_new]
        self.gains_ = {n: round(gains[n], 5) for n in chosen}
        self.specs_ = {n: cands[n] for n in chosen}
        self.tables_ = {}
        for n, spec in self.specs_.items():
            if spec[0] == "group_mean":
                _, g, v = spec
                key = df[g].astype(str).where(df[g].notna(), "__nan__")
                self.tables_[n] = (df[v].astype(float).groupby(key).mean(), float(df[v].astype(float).mean()))
            ctx.types[n] = ColumnType.NUMERIC
        self.note(
            f"{len(cands)} candidates -> {len(stage1)} after MI screen -> {len(chosen)} kept by residual boosting"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new = {}
        for n, spec in self.specs_.items():
            cols = spec[2:] if spec[0] == "op" else spec[1:]
            if all(c in df.columns for c in cols):
                new[n] = self._compute(df, spec, self.tables_.get(n))
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df.copy()
