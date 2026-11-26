"""Categorical encoders: one-hot (top-k + other), frequency, ordinal, and leakage-safe target encoding."""

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..types import ColumnType, parse_numeric
from .base import FitContext, Step, StepError, register

OTHER = "__other__"
NAN = "__nan__"
MAX_TE_CLASSES = 20


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _key(s: pd.Series) -> pd.Series:
    return s.astype(str).where(s.notna(), NAN)


def _cat_columns(df: pd.DataFrame, ctx: FitContext, requested: Optional[List[str]]) -> List[str]:
    return ctx.columns_of(df, (ColumnType.CATEGORICAL,), requested)


# ---------------------------------------------------------------------------------------------- one-hot
class OneHotParams(_P):
    columns: Optional[List[str]] = None
    max_categories: int = Field(20, ge=1, le=500, description="keep at most this many categories per column")
    min_frequency: float = Field(0.01, ge=0, lt=1, description="rarer categories go to __other__")
    max_cardinality: Optional[int] = Field(
        None, ge=2, description="only encode columns with at most this many distinct values (auto selection)"
    )
    drop_original: bool = True


@register
class OneHot(Step):
    name = "one_hot"
    kind = "encode"
    description = "One-hot encode categories (top-k, rare and unseen values go to an __other__ column)."
    Params = OneHotParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "OneHot":
        p = self.params
        cols = _cat_columns(df, ctx, p.columns)
        if p.max_cardinality is not None and p.columns is None:
            cols = [c for c in cols if df[c].nunique() <= p.max_cardinality]
        self.levels_: Dict[str, List[str]] = {}
        for c in cols:
            freq = _key(df[c]).value_counts(normalize=True)
            keep = [str(v) for v, f in freq.items() if f >= p.min_frequency][: p.max_categories]
            self.levels_[c] = keep
            for name in self._names(c):
                ctx.types[name] = ColumnType.BOOLEAN
            if p.drop_original:
                ctx.types.pop(c, None)
        self.note(f"{sum(len(self._names(c)) for c in cols)} indicator column(s) from {len(cols)} column(s)")
        return self

    def _names(self, c: str) -> List[str]:
        return [f"{c}={v}" for v in self.levels_[c]] + [f"{c}={OTHER}"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        blocks = []
        for c, levels in self.levels_.items():
            if c not in df.columns:
                continue
            k = _key(df[c])
            block = {f"{c}={v}": (k == v).astype("int8") for v in levels}
            block[f"{c}={OTHER}"] = (~k.isin(levels)).astype("int8")
            blocks.append(pd.DataFrame(block, index=df.index))
        if blocks:
            out = pd.concat([df] + blocks, axis=1)
        if self.params.drop_original:
            out = out.drop(columns=[c for c in self.levels_ if c in out.columns])
        return out


# ------------------------------------------------------------------------------------------- frequency
class FrequencyParams(_P):
    columns: Optional[List[str]] = None
    normalize: bool = True
    min_cardinality: Optional[int] = Field(None, ge=1, description="auto selection: only columns with more values")
    drop_original: bool = False


@register
class FrequencyEncode(Step):
    name = "frequency_encode"
    kind = "encode"
    description = "Replace each category by how often it occurs in the training data (unseen -> 0)."
    Params = FrequencyParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "FrequencyEncode":
        p = self.params
        cols = _cat_columns(df, ctx, p.columns)
        if p.min_cardinality is not None and p.columns is None:
            cols = [c for c in cols if df[c].nunique() > p.min_cardinality]
        self.maps_ = {c: _key(df[c]).value_counts(normalize=p.normalize).to_dict() for c in cols}
        for c in cols:
            ctx.types[f"{c}__freq"] = ColumnType.NUMERIC
            if p.drop_original:
                ctx.types.pop(c, None)
        self.note(f"frequency-encoded {len(cols)} column(s)")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new = {f"{c}__freq": _key(df[c]).map(m).astype(float).fillna(0.0) for c, m in self.maps_.items() if c in df}
        out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df.copy()
        if self.params.drop_original:
            out = out.drop(columns=[c for c in self.maps_ if c in out.columns])
        return out


# --------------------------------------------------------------------------------------------- ordinal
class OrdinalParams(_P):
    columns: Optional[List[str]] = None
    order: Literal["frequency", "lexical"] = "frequency"


@register
class OrdinalEncode(Step):
    name = "ordinal_encode"
    kind = "encode"
    description = "Replace categories by integer codes in place (unseen -> -1); compact input for tree models."
    Params = OrdinalParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "OrdinalEncode":
        cols = _cat_columns(df, ctx, self.params.columns)
        self.maps_: Dict[str, Dict[str, int]] = {}
        for c in cols:
            k = _key(df[c])
            levels = k.value_counts().index.tolist() if self.params.order == "frequency" else sorted(k.unique())
            self.maps_[c] = {v: i for i, v in enumerate(levels)}
            ctx.types[c] = ColumnType.NUMERIC
        self.note(f"ordinal-encoded {len(cols)} column(s)")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, m in self.maps_.items():
            if c in out.columns:
                out[c] = _key(out[c]).map(m).fillna(-1).astype("int64")
        return out


# -------------------------------------------------------------------------------------- target encoding
class TargetEncodeParams(_P):
    columns: Optional[List[str]] = None
    smoothing: float = Field(10.0, ge=0, description="m in (n*mean + m*prior)/(n + m)")
    n_folds: int = Field(5, ge=2, le=20)
    scheme: Literal["oof", "ordered"] = Field("oof", description="how training rows are encoded")
    min_cardinality: Optional[int] = Field(None, ge=1, description="auto selection: only columns with more values")
    drop_original: bool = True


@register
class TargetEncode(Step):
    name = "target_encode"
    kind = "encode"
    description = (
        "Smoothed target mean per category (Micci-Barreca 2001). Training rows get out-of-fold or ordered "
        "(CatBoost-style) values so the encoding does not leak the row's own target."
    )
    Params = TargetEncodeParams
    needs_target = True

    # ---- target matrix
    def _target_matrix(self, y: pd.Series, ctx: FitContext) -> pd.DataFrame:
        if ctx.task == "regression":
            return pd.DataFrame({"te": parse_numeric(y)}, index=y.index)
        classes = ctx.classes or sorted(y.dropna().astype(str).unique())
        ys = y.astype(str).where(y.notna())
        if len(classes) <= 2:
            pos = classes[-1]
            return pd.DataFrame({"te": (ys == pos).astype(float).where(ys.notna())}, index=y.index)
        if len(classes) > MAX_TE_CLASSES:
            raise StepError(f"target_encode supports up to {MAX_TE_CLASSES} classes, got {len(classes)}")
        return pd.DataFrame(
            {f"te_{k}": (ys == k).astype(float).where(ys.notna()) for k in classes},
            index=y.index,
        )

    def _stats(self, key: pd.Series, Y: pd.DataFrame) -> tuple:
        ok = Y.iloc[:, 0].notna()
        g = Y[ok].groupby(key[ok])
        return g.sum(), g.count(), Y[ok].mean()

    def _encode(self, key: pd.Series, sums: pd.DataFrame, counts: pd.DataFrame, prior: pd.Series) -> pd.DataFrame:
        m = self.params.smoothing
        out = {}
        for col in prior.index:
            s = key.map(sums[col]).astype(float).fillna(0.0)
            n = key.map(counts[col]).astype(float).fillna(0.0)
            out[col] = (s + m * prior[col]) / (n + m) if m > 0 else (s / n.where(n > 0)).fillna(prior[col])
        return pd.DataFrame(out, index=key.index)

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "TargetEncode":
        y = ctx.y(df)
        if y is None:
            raise StepError("target_encode needs the target column in the training data")
        if ctx.task not in ("classification", "regression"):
            raise StepError("target_encode needs task classification or regression")
        p = self.params
        cols = _cat_columns(df, ctx, p.columns)
        if p.min_cardinality is not None and p.columns is None:
            cols = [c for c in cols if df[c].nunique() > p.min_cardinality]
        self.columns_ = cols
        Y = self._target_matrix(y, ctx)
        self.target_cols_ = list(Y.columns)
        self.tables_ = {c: self._stats(_key(df[c]), Y) for c in cols}
        for c in cols:
            for name in self._names(c):
                ctx.types[name] = ColumnType.NUMERIC
            if p.drop_original:
                ctx.types.pop(c, None)
        self.note(f"target-encoded {len(cols)} column(s) ({p.scheme}, smoothing {p.smoothing})")
        self._ctx_task = ctx.task
        return self

    def _names(self, c: str) -> List[str]:
        return [f"{c}__{t}" for t in self.target_cols_]

    def fit_transform(self, df: pd.DataFrame, ctx: FitContext) -> pd.DataFrame:
        self.fit(df, ctx)
        self.fitted_ = True
        y = ctx.y(df)
        Y = self._target_matrix(y, ctx)
        blocks = []
        for c in self.columns_:
            key = _key(df[c])
            enc = self._oof(key, Y, ctx) if self.params.scheme == "oof" else self._ordered(key, Y, ctx)
            enc.columns = self._names(c)
            blocks.append(enc)
        return self._assemble(df, blocks)

    def _oof(self, key: pd.Series, Y: pd.DataFrame, ctx: FitContext) -> pd.DataFrame:
        from sklearn.model_selection import KFold, StratifiedKFold

        n = len(key)
        result = pd.DataFrame(np.nan, index=key.index, columns=Y.columns)
        labels = Y.idxmax(axis=1) if Y.shape[1] > 1 else Y.iloc[:, 0]
        ok = labels.notna().to_numpy()
        use_strat = ctx.task == "classification" and labels[ok].value_counts().min() >= self.params.n_folds
        splitter = (
            StratifiedKFold(self.params.n_folds, shuffle=True, random_state=ctx.random_state)
            if use_strat
            else KFold(self.params.n_folds, shuffle=True, random_state=ctx.random_state)
        )
        strat = labels.astype(str).to_numpy() if use_strat else np.zeros(n)
        for fit_idx, enc_idx in splitter.split(np.zeros(n), strat):
            sums, counts, prior = self._stats(key.iloc[fit_idx], Y.iloc[fit_idx])
            prior = prior.fillna(Y.mean())
            result.iloc[enc_idx] = self._encode(key.iloc[enc_idx], sums, counts, prior).to_numpy()
        return result

    def _ordered(self, key: pd.Series, Y: pd.DataFrame, ctx: FitContext) -> pd.DataFrame:
        rng = np.random.default_rng(ctx.random_state)
        perm = rng.permutation(len(key))
        k = key.iloc[perm].reset_index(drop=True)
        Yp = Y.iloc[perm].reset_index(drop=True)
        prior = Y.mean()
        m = self.params.smoothing
        out = {}
        for col in Y.columns:
            v = Yp[col]
            filled = v.fillna(0.0)
            prev_sum = filled.groupby(k).cumsum() - filled
            prev_n = v.notna().astype(float).groupby(k).cumsum() - v.notna().astype(float)
            out[col] = ((prev_sum + m * prior[col]) / (prev_n + m)) if m > 0 else (prev_sum / prev_n).fillna(prior[col])
        enc = pd.DataFrame(out)
        enc.index = key.index[perm]
        return enc.loc[key.index]

    def _assemble(self, df: pd.DataFrame, blocks: List[pd.DataFrame]) -> pd.DataFrame:
        out = pd.concat([df] + blocks, axis=1) if blocks else df.copy()
        if self.params.drop_original:
            out = out.drop(columns=[c for c in self.columns_ if c in out.columns])
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        blocks = []
        for c in self.columns_:
            if c not in df.columns:
                continue
            sums, counts, prior = self.tables_[c]
            enc = self._encode(_key(df[c]), sums, counts, prior)
            enc.columns = self._names(c)
            blocks.append(enc)
        return self._assemble(df, blocks)
