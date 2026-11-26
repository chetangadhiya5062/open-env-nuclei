"""FeatureForge: type inference, IO, profiling, drift and splitting."""

import io

import numpy as np
import pandas as pd
import pytest

from featureforge import ColumnType, drift_report, infer_types, leakage_safe_split, profile, read_table, write_table
from featureforge.io import TableReadError
from featureforge.samples import TARGETS, make_dataset
from featureforge.types import parse_bool, parse_datetime, parse_numeric


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("FF_WORK_DIR", str(tmp_path / "ws"))


# ------------------------------------------------------------------------------------------------ types
def test_parse_numeric_handles_text_formats():
    s = pd.Series(["1,250", "$12.50", "45%", "(3)", "N/A", None, 7, "abc"])
    out = parse_numeric(s).tolist()
    assert out[:5] == [1250.0, 12.5, 45.0, -3.0, out[4]] and np.isnan(out[4])
    assert np.isnan(out[5]) and out[6] == 7.0 and np.isnan(out[7])


def test_parse_bool_and_dates():
    assert parse_bool(pd.Series(["yes", "N", "TRUE", "0", "maybe"])).tolist()[:4] == [1.0, 0.0, 1.0, 0.0]
    d = parse_datetime(pd.Series(["2024-01-05", "5 Jan 2024", "Jan 05, 2024", "2024/01/05", "nope"]))
    assert d.iloc[:4].nunique() == 1 and pd.isna(d.iloc[4])


def test_infer_types_covers_every_kind():
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "amount": [f"${x:,.2f}" for x in rng.uniform(1, 5000, n)],
            "score": rng.normal(size=n),
            "city": rng.choice(["a", "b", "c"], n),
            "joined": pd.date_range("2024-01-01", periods=n).strftime("%d/%m/%Y"),
            "active": rng.choice(["yes", "no", "Y"], n),
            "comment": [f"this is a fairly long free text comment number {i}" for i in range(n)],
            "const": 1,
        }
    )
    t = infer_types(df)
    assert t == {
        "user_id": ColumnType.ID,
        "amount": ColumnType.NUMERIC,
        "score": ColumnType.NUMERIC,
        "city": ColumnType.CATEGORICAL,
        "joined": ColumnType.DATETIME,
        "active": ColumnType.BOOLEAN,
        "comment": ColumnType.TEXT,
        "const": ColumnType.CONSTANT,
    }
    assert infer_types(df, {"score": "categorical"})["score"] == ColumnType.CATEGORICAL


# ------------------------------------------------------------------------------------------------- io
@pytest.mark.parametrize("ext", [".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".feather"])
def test_roundtrip_formats(tmp_path, ext):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None], "c": [1.5, None, 3.0]})
    path = write_table(df, tmp_path / f"t{ext}")
    back = read_table(path)
    assert list(back.columns) == ["a", "b", "c"] and len(back) == 3
    assert back["a"].tolist() == [1, 2, 3]


def test_csv_delimiter_encoding_and_duplicate_headers():
    data = "name;value;value\nJosé;1;2\nZoë;3;4\n".encode("cp1252")
    df = read_table(data, name="x.csv")
    assert list(df.columns) == ["name", "value", "value.1"] and df["name"].tolist() == ["José", "Zoë"]


@pytest.mark.parametrize(
    "data,name,msg",
    [
        (b"", "a.csv", "empty"),
        (b"x", "a.xls", "xls"),
        (b"x", "a.docx", "unsupported"),
        (b"\x00garbage", "a.parquet", "parquet"),
    ],
)
def test_read_errors_are_readable(data, name, msg):
    with pytest.raises(TableReadError, match=msg):
        read_table(io.BytesIO(data), name=name)


# --------------------------------------------------------------------------------------------- profile
def test_profile_flags_real_problems():
    df = make_dataset("churn", 1500, 0)
    p = profile(df, target="churned")
    codes = {(w["code"], w["column"]) for w in p["warnings"]}
    assert ("possible_target_leak", "cancellation_reason") in codes
    assert ("constant", "constant_col") in codes
    assert ("id_column", "customer_id") in codes
    assert ("numeric_as_text", "monthly_spend") in codes
    assert ("inconsistent_categories", "city") in codes
    assert ("duplicates", None) in codes
    assert p["target"]["task"] == "classification"
    assert 0 <= p["quality_score"] <= 100
    assert p["dataset"]["duplicate_rows"] == 45


def test_profile_samples_large_inputs(monkeypatch):
    monkeypatch.setenv("FF_PROFILE_SAMPLE_ROWS", "300")
    p = profile(make_dataset("housing", 1000, 1))
    assert p["dataset"]["sampled"] and p["dataset"]["sample_rows"] == 300
    assert p["dataset"]["rows"] == 1020  # full row count is still reported


def test_profile_detects_target_copy_and_unknown_target():
    df = pd.DataFrame({"x": range(100), "y": [i % 2 for i in range(100)]})
    df["y_copy"] = df["y"]
    p = profile(df, target="y")
    assert any(w["code"] == "target_copy" and w["column"] == "y_copy" for w in p["warnings"])
    with pytest.raises(KeyError):
        profile(df, target="nope")


def test_drift_report_distinguishes_stable_and_shifted():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(size=3000), "c": rng.choice(["x", "y"], 3000)})
    same = pd.DataFrame({"a": rng.normal(size=3000), "c": rng.choice(["x", "y"], 3000)})
    moved = pd.DataFrame({"a": rng.normal(2, 1, size=3000), "c": rng.choice(["x", "z"], 3000), "new": 1})
    assert drift_report(ref, same)["drifted"] == []
    rep = drift_report(ref, moved)
    assert set(rep["drifted"]) == {"a", "c"} and rep["new_columns"] == ["new"]


# ------------------------------------------------------------------------------------------------ split
def test_split_never_shares_duplicate_rows():
    df = make_dataset("churn", 1000, 2)
    train, test, rep = leakage_safe_split(df, target="churned", test_size=0.25)
    assert rep["strategy"] == "stratified" and rep["shared_feature_rows"] == 0
    feats = [c for c in df.columns if c != "churned"]
    h = lambda d: set(pd.util.hash_pandas_object(d[feats].astype(str), index=False))  # noqa: E731
    assert not (h(train) & h(test))
    assert len(train) + len(test) == len(df)


def test_group_and_time_split():
    df = pd.DataFrame(
        {
            "user": np.repeat(np.arange(50), 4),
            "t": pd.date_range("2024-01-01", periods=200).astype(str),
            "v": range(200),
        }
    )
    tr, te, rep = leakage_safe_split(df, group_column="user", test_size=0.2)
    assert rep["shared_groups"] == 0
    tr, te, rep = leakage_safe_split(df, time_column="t", test_size=0.2)
    assert rep["strategy"] == "time" and pd.to_datetime(tr["t"]).max() < pd.to_datetime(te["t"]).min()
    with pytest.raises(ValueError):
        leakage_safe_split(df, test_size=1.5)


@pytest.mark.parametrize("kind", ["churn", "housing", "transactions"])
def test_samples_are_deterministic(kind):
    a, b = make_dataset(kind, 300, 5), make_dataset(kind, 300, 5)
    pd.testing.assert_frame_equal(a, b)
    assert TARGETS[kind] in a.columns
