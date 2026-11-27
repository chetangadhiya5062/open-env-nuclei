"""FeatureForge: REST API, CLI, dashboard handlers, HTML report and dataset card."""

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from featureforge.cli import app as cli_app
from featureforge.samples import make_dataset


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("FF_WORK_DIR", str(tmp_path / "ws"))
    monkeypatch.delenv("FF_API_KEY", raising=False)


@pytest.fixture(scope="module")
def client():
    from featureforge.api import app

    return TestClient(app)


def _csv(kind="churn", rows=300, seed=0) -> bytes:
    return make_dataset(kind, rows, seed).to_csv(index=False).encode()


# ----------------------------------------------------------------------------------------------------- API
def test_health_info_steps(client):
    assert client.get("/health").json()["status"] == "ok"
    assert ".parquet" in client.get("/v1/info").json()["formats"]
    assert any(s["step"] == "target_encode" for s in client.get("/v1/steps").json())


def test_profile_and_auto_recipe(client):
    r = client.post("/v1/profile", files={"file": ("d.csv", _csv())}, data={"target": "churned"})
    assert r.status_code == 200 and r.json()["target"]["task"] == "classification"
    r = client.post("/v1/recipes/auto", files={"file": ("d.csv", _csv())}, data={"target": "churned", "level": "full"})
    assert r.status_code == 200 and "feature_generation" in r.json()["yaml"]
    v = client.post("/v1/recipes/validate", json=r.json()["recipe"]).json()
    assert v == {"valid": True, "errors": []}
    v = client.post("/v1/recipes/validate", json={"steps": [{"step": "nope"}]}).json()
    assert not v["valid"]


def test_run_download_and_transform(client):
    r = client.post(
        "/v1/runs",
        files={"file": ("churn.csv", _csv(rows=400))},
        data={"target": "churned", "test_size": "0.2"},
    )
    assert r.status_code == 200, r.text
    s = r.json()
    run_id = s["run_id"]
    assert s["split"]["shared_feature_rows"] == 0 and s["test_output_shape"][1] == s["output_shape"][1]
    assert client.get(f"/v1/runs/{run_id}").json()["run_id"] == run_id
    assert any(x["run_id"] == run_id for x in client.get("/v1/runs").json())
    for name in ("data.csv", "data.parquet", "report.html", "datacard.md", "recipe.yaml", "bundle.zip"):
        f = client.get(f"/v1/runs/{run_id}/files/{name}")
        assert f.status_code == 200 and len(f.content) > 100, name
    assert client.get(f"/v1/runs/{run_id}/files/..%2Fsecret").status_code == 404
    assert client.get("/v1/runs/../files/x").status_code == 404
    new_df = make_dataset("churn", 50, 7).drop(columns=["churned"])
    new = new_df.to_csv(index=False).encode()
    t = client.post(f"/v1/runs/{run_id}/transform", files={"file": ("new.csv", new)}, data={"format": "csv"})
    assert t.status_code == 200
    out = pd.read_csv(io.BytesIO(t.content))
    assert len(out) == len(new_df) and out.shape[1] == s["output_shape"][1] - 1


def test_run_with_explicit_recipe(client):
    recipe = "steps:\n  - step: normalize_strings\n  - step: fix_types\n  - step: impute\n"
    r = client.post("/v1/runs", files={"file": ("h.csv", _csv("housing"))}, data={"recipe": recipe})
    assert r.status_code == 200 and len(r.json()["pipeline"]["steps"]) == 3
    bad = client.post("/v1/runs", files={"file": ("h.csv", _csv("housing"))}, data={"recipe": "steps: [{step: x}]"})
    assert bad.status_code == 422


def test_api_errors(client, monkeypatch):
    assert client.post("/v1/profile", files={"file": ("a.docx", b"x")}).status_code == 415
    assert client.post("/v1/profile", files={"file": ("a.csv", b"")}).status_code == 422
    r = client.post("/v1/profile", files={"file": ("a.csv", _csv())}, data={"target": "missing_col"})
    assert r.status_code == 422
    assert client.get("/v1/runs/zzzzzzzzzzzz").status_code == 404
    monkeypatch.setenv("FF_MAX_UPLOAD_MB", "0")
    assert client.post("/v1/profile", files={"file": ("a.csv", _csv())}).status_code == 413


def test_api_key(client, monkeypatch):
    monkeypatch.setenv("FF_API_KEY", "s3cret")
    assert client.get("/v1/info").status_code == 401
    assert client.get("/v1/info", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/info", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200


# ----------------------------------------------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path):
    runner = CliRunner()
    data = tmp_path / "d.csv"
    r = runner.invoke(cli_app, ["sample", "churn", "--rows", "300", "--out", str(data)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        cli_app,
        ["profile", str(data), "-t", "churned", "-o", str(tmp_path / "r.html"), "--json", str(tmp_path / "p.json")],
    )
    assert r.exit_code == 0 and "quality score" in r.output
    assert json.loads((tmp_path / "p.json").read_text())["dataset"]["rows"] > 0
    r = runner.invoke(cli_app, ["auto", str(data), "-t", "churned", "-l", "full", "-o", str(tmp_path / "rec.yaml")])
    assert r.exit_code == 0
    assert runner.invoke(cli_app, ["validate-recipe", str(tmp_path / "rec.yaml")]).exit_code == 0
    out = tmp_path / "out"
    r = runner.invoke(
        cli_app, ["run", str(data), "--recipe", str(tmp_path / "rec.yaml"), "-o", str(out), "--test-size", "0.2"]
    )
    assert r.exit_code == 0, r.output
    assert (
        (out / "data.parquet").exists()
        and (out / "test.csv").exists()
        and (out / "pipeline" / "pipeline.joblib").exists()
    )
    new = tmp_path / "new.csv"
    make_dataset("churn", 40, 3).drop(columns=["churned"]).to_csv(new, index=False)
    r = runner.invoke(cli_app, ["transform", str(new), "-p", str(out / "pipeline"), "-o", str(tmp_path / "t.parquet")])
    assert r.exit_code == 0 and len(pd.read_parquet(tmp_path / "t.parquet")) == len(pd.read_csv(new))
    r = runner.invoke(cli_app, ["split", str(data), "-t", "churned", "-o", str(tmp_path / "sp")])
    assert r.exit_code == 0 and (tmp_path / "sp" / "train.csv").exists()
    r = runner.invoke(cli_app, ["drift", str(data), str(new)])
    assert r.exit_code == 0
    assert "target_encode" in runner.invoke(cli_app, ["steps"]).output


def test_cli_errors(tmp_path):
    runner = CliRunner()
    r = runner.invoke(cli_app, ["profile", str(tmp_path / "missing.csv")])
    assert r.exit_code == 1 and "file not found" in r.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("steps:\n  - step: wizardry\n")
    r = runner.invoke(cli_app, ["validate-recipe", str(bad)])
    assert r.exit_code == 1 and "unknown step" in r.output


# ----------------------------------------------------------------------------------------------- dashboard
def test_dashboard_handlers(tmp_path):
    from featureforge import dashboard as d

    st, msg, prev, target = d.load_sample("housing", 400, d._empty_state())
    assert "Loaded" in msg and target == "price" and len(prev) > 0
    res = d.do_profile(st, "price")
    st = res[0]
    assert "Quality score" in res[1] and len(res[2]) > 0 and res[-1]
    fig, detail = d.column_detail(st, "sqft")
    assert fig is not None and "numeric" in detail
    msg, yaml_text, names = d.make_recipe(st, "price", "full")
    assert names and "steps" in yaml_text
    assert "valid" in d.validate_yaml(yaml_text)
    out = d.do_run(st, yaml_text, names[:-1], "price", 0.2, "auto", None, None)
    st = out[0]
    assert "finished" in out[1] and len(out[2]) == len(names) - 1
    assert "Before" in d.before_after(st, "sqft")
    files, card = d.export_files(st)
    assert any(f.endswith("data.parquet") for f in files) and "Dataset card" in card
    new = tmp_path / "new.csv"
    make_dataset("housing", 30, 9).to_csv(new, index=False)
    path, msg, prev = d.apply_new(st, str(new), "parquet")
    assert path.endswith(".parquet") and len(prev) == 20
    st2, msg, _, _ = d.load_file(None, d._empty_state())
    assert "Error" in msg


def test_dashboard_builds():
    from featureforge.dashboard import build_dashboard

    assert build_dashboard() is not None


# ------------------------------------------------------------------------------------------ report & card
def test_report_and_datacard_content():
    from featureforge import Pipeline, auto_recipe, profile
    from featureforge.datacard import build_datacard
    from featureforge.report import render_report

    df = make_dataset("churn", 300, 0)
    p = profile(df, target="churned")
    pipe = Pipeline(auto_recipe(df, "churned"))
    out = pipe.fit_transform(df)
    html = render_report(p, steps=pipe.summary()["steps"], preview=out)
    for needle in ("Quality score", "Warnings", "possible_target_leak", "Pipeline", "data:image/png;base64"):
        assert needle in html
    assert "<script" not in html.lower()
    card = build_datacard("churn.csv", p, out, pipe.summary(), pipe.recipe.to_yaml())
    for needle in ("## Motivation", "## Composition", "## Preprocessing", "To be completed", "deduplicate"):
        assert needle in card


def test_leak_policy_in_api_cli_report(client, tmp_path):
    r = client.post(
        "/v1/runs", files={"file": ("c.csv", _csv(rows=400))}, data={"target": "churned", "leak_policy": "warn"}
    )
    assert r.status_code == 200
    s = r.json()
    assert s["leak_policy"] == "warn"
    assert any(x["column"] == "cancellation_reason" and x["action"] == "kept (warning)" for x in s["leaks"])
    html = client.get(f"/v1/runs/{s['run_id']}/files/report.html").text
    assert "Target leak check" in html and "cancellation_reason" in html
    card = client.get(f"/v1/runs/{s['run_id']}/files/datacard.md").text
    assert "Target-leak check" in card
    bad = client.post("/v1/runs", files={"file": ("c.csv", _csv())}, data={"leak_policy": "nope"})
    assert bad.status_code == 422
    r = client.post(
        "/v1/recipes/auto", files={"file": ("c.csv", _csv())}, data={"target": "churned", "leak_policy": "drop"}
    )
    assert "leak_policy: drop" in r.json()["yaml"]

    runner = CliRunner()
    data = tmp_path / "d.csv"
    make_dataset("churn", 300, 0).to_csv(data, index=False)
    res = runner.invoke(
        cli_app, ["run", str(data), "-t", "churned", "--leak-policy", "auto", "-o", str(tmp_path / "o")]
    )
    assert res.exit_code == 0 and "target leak: cancellation_reason" in res.output and "dropped" in res.output
    res = runner.invoke(cli_app, ["auto", str(data), "-t", "churned", "--leak-policy", "bogus"])
    assert res.exit_code == 1
