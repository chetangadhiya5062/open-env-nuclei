"""Command line interface: ``fe`` (alias ``featureforge``).

fe profile data.csv --target churned --out report.html
fe auto data.csv --target churned --level standard --out recipe.yaml
fe run data.csv --recipe recipe.yaml --out-dir out/          # or without --recipe: auto mode
fe run data.csv --target churned --test-size 0.2             # leakage-safe split, fit on train only
fe transform new.csv --pipeline out/pipeline --out new_features.parquet
fe drift train.csv new.csv
fe steps                                                     # list every step and its parameters
fe serve --port 8000                                         # REST API + dashboard at /ui
"""

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from .log import setup_logging

app = typer.Typer(add_completion=False, help="FeatureForge: data cleaning and feature engineering for tables.")
logger = logging.getLogger("featureforge.cli")


def _fail(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _read(path: Path):
    from .io import TableReadError, read_table

    if not path.exists():
        _fail(f"file not found: {path}")
    try:
        return read_table(path)
    except TableReadError as exc:
        _fail(str(exc))


def _check_policy(policy: str) -> None:
    if policy not in ("auto", "drop", "warn"):
        _fail("--leak-policy must be auto, drop or warn")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="debug logging")) -> None:
    setup_logging("DEBUG" if verbose else None)


@app.command()
def profile(
    data: Path = typer.Argument(..., help="CSV, Excel, Parquet or JSON file"),
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    out: Path = typer.Option(Path("report.html"), "--out", "-o", help="HTML report path"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="also write the profile as JSON"),
) -> None:
    """Profile a table and write an HTML data-quality report."""
    from .profile import profile as run_profile
    from .report import write_report

    df = _read(data)
    try:
        p = run_profile(df, target=target)
    except KeyError as exc:
        _fail(str(exc))
    write_report(out, p, title=f"Data quality report: {data.name}", source=data.name)
    if json_out:
        json_out.write_text(json.dumps(p, indent=1, default=str), encoding="utf-8")
    crit = sum(w["severity"] == "critical" for w in p["warnings"])
    typer.echo(f"quality score {p['quality_score']}/100, {len(p['warnings'])} warning(s) ({crit} critical) -> {out}")


@app.command()
def auto(
    data: Path = typer.Argument(...),
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    level: str = typer.Option("standard", "--level", "-l", help="clean | standard | full"),
    out: Path = typer.Option(Path("recipe.yaml"), "--out", "-o"),
    leak_policy: str = typer.Option(
        "auto", "--leak-policy", help="auto (drop high-confidence target leaks, warn on the rest) | drop | warn"
    ),
) -> None:
    """Generate a recipe from the data (edit it, then use `fe run --recipe`)."""
    from .auto import auto_recipe

    if level not in ("clean", "standard", "full"):
        _fail("level must be clean, standard or full")
    _check_policy(leak_policy)
    df = _read(data)
    try:
        r = auto_recipe(df, target=target, level=level, name=data.stem, leak_policy=leak_policy)  # type: ignore[arg-type]
    except KeyError as exc:
        _fail(str(exc))
    r.save(out)
    typer.echo(f"{len(r.steps)} step(s): {', '.join(s.step for s in r.steps)} -> {out}")


@app.command()
def run(
    data: Path = typer.Argument(...),
    recipe: Optional[Path] = typer.Option(None, "--recipe", "-r", help="recipe YAML/JSON; omit for auto mode"),
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    level: str = typer.Option("standard", "--level", "-l"),
    out_dir: Path = typer.Option(Path("ff_output"), "--out-dir", "-o"),
    test_size: Optional[float] = typer.Option(None, "--test-size", help="leakage-safe split; fit on train only"),
    split_strategy: str = typer.Option("auto", "--split-strategy", help="auto | random | stratified | group | time"),
    group_column: Optional[str] = typer.Option(None, "--group-column"),
    time_column: Optional[str] = typer.Option(None, "--time-column"),
    leak_policy: str = typer.Option(
        "auto", "--leak-policy", help="auto (drop high-confidence target leaks, warn on the rest) | drop | warn"
    ),
) -> None:
    """Clean + engineer features, and write the dataset, pipeline, report and dataset card."""
    from .pipeline import PipelineError
    from .recipe import Recipe
    from .service import run_job

    _check_policy(leak_policy)
    df = _read(data)
    rec = None
    if recipe is not None:
        if not recipe.exists():
            _fail(f"recipe not found: {recipe}")
        try:
            rec = Recipe.from_file(recipe)
        except Exception as exc:
            _fail(f"invalid recipe: {exc}")
    try:
        res = run_job(
            df,
            out_dir=out_dir,
            recipe=rec,
            target=target,
            level=level,
            test_size=test_size,
            split_strategy=split_strategy,
            group_column=group_column,
            time_column=time_column,
            source_name=data.name,
            source_path=data,
            leak_policy=leak_policy,
        )
    except (PipelineError, KeyError, ValueError) as exc:
        _fail(str(exc))
    s = res.summary
    typer.echo(f"{tuple(s['input_shape'])} -> {tuple(s['output_shape'])} in {s['seconds']}s; task {s['task']}")
    if s["test_output_shape"]:
        typer.echo(f"test part: {tuple(s['test_output_shape'])} (split {s['split']['strategy']})")
    for leak in s.get("leaks") or []:
        typer.secho(
            f"target leak: {leak['column']} ({leak['code']}, {leak['confidence']} confidence) -> {leak['action']}",
            fg=typer.colors.YELLOW,
        )
    for step in s["pipeline"]["steps"]:
        typer.echo(f"  {step['index']:>2}. {step['step']:<20} {step['cols_before']:>4} -> {step['cols_after']:<4} cols")
    typer.echo(f"artifacts in {out_dir}: {', '.join(sorted(res.files))}")


@app.command()
def transform(
    data: Path = typer.Argument(...),
    pipeline: Path = typer.Option(..., "--pipeline", "-p", help="folder written by `fe run` (…/pipeline)"),
    out: Path = typer.Option(Path("transformed.csv"), "--out", "-o", help=".csv / .parquet / .json / .xlsx"),
) -> None:
    """Apply a saved, fitted pipeline to new data (only load pipelines you created)."""
    from .io import write_table
    from .pipeline import Pipeline, PipelineError

    df = _read(data)
    try:
        pipe = Pipeline.load(pipeline)
        res = pipe.transform(df)
    except PipelineError as exc:
        _fail(str(exc))
    for w in pipe.warnings_:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW, err=True)
    write_table(res, out)
    typer.echo(f"{df.shape} -> {res.shape} -> {out}")


@app.command()
def split(
    data: Path = typer.Argument(...),
    target: Optional[str] = typer.Option(None, "--target", "-t"),
    test_size: float = typer.Option(0.2, "--test-size"),
    strategy: str = typer.Option("auto", "--strategy"),
    group_column: Optional[str] = typer.Option(None, "--group-column"),
    time_column: Optional[str] = typer.Option(None, "--time-column"),
    out_dir: Path = typer.Option(Path("split"), "--out-dir", "-o"),
) -> None:
    """Leakage-safe train/test split (duplicates and groups never straddle the split)."""
    from .io import write_table
    from .split import leakage_safe_split

    df = _read(data)
    try:
        tr, te, rep = leakage_safe_split(df, target, test_size, strategy, group_column, time_column)  # type: ignore[arg-type]
    except (KeyError, ValueError) as exc:
        _fail(str(exc))
    write_table(tr, out_dir / f"train{data.suffix if data.suffix in ('.csv', '.parquet') else '.csv'}")
    write_table(te, out_dir / f"test{data.suffix if data.suffix in ('.csv', '.parquet') else '.csv'}")
    (out_dir / "split_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    typer.echo(json.dumps(rep, indent=1))


@app.command()
def drift(
    reference: Path = typer.Argument(..., help="training / reference table"),
    current: Path = typer.Argument(..., help="new table"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="write the drift report as JSON"),
) -> None:
    """Compare a new table with a reference table (PSI + KS test per column)."""
    from .profile import drift_report

    rep = drift_report(_read(reference), _read(current))
    if out:
        out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    for e in rep["columns"]:
        typer.echo(f"{e['column']:<30} psi={e['psi']:<8} {e['status']}")
    if rep["missing_columns"]:
        typer.secho(f"missing columns: {rep['missing_columns']}", fg=typer.colors.YELLOW)


@app.command()
def steps() -> None:
    """List every pipeline step with its description."""
    from .steps import catalog

    for s in catalog():
        flags = (" [train-only]" if s["train_only"] else "") + (" [needs target]" if s["needs_target"] else "")
        typer.echo(f"{s['step']:<20} {s['kind']:<8} {s['description']}{flags}")


@app.command("validate-recipe")
def validate_recipe(recipe: Path = typer.Argument(...)) -> None:
    """Check a recipe file for unknown steps and invalid parameters."""
    from .recipe import Recipe

    try:
        r = Recipe.from_file(recipe)
    except Exception as exc:
        _fail(f"invalid recipe: {exc}")
    errors = r.validate_steps()
    if errors:
        _fail("\n".join(errors))
    typer.echo(f"ok: {len(r.steps)} step(s)")


@app.command()
def sample(
    kind: str = typer.Argument("churn", help="churn | housing | transactions"),
    rows: int = typer.Option(2000, "--rows"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("sample.csv"), "--out", "-o"),
) -> None:
    """Write a synthetic messy dataset to try the tool on."""
    from .io import write_table
    from .samples import make_dataset

    try:
        df = make_dataset(kind, rows, seed)
    except ValueError as exc:
        _fail(str(exc))
    write_table(df, out)
    typer.echo(f"{df.shape} -> {out}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the REST API with the dashboard mounted at /ui."""
    import uvicorn

    uvicorn.run("featureforge.api:app", host=host, port=port)


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7861, "--port"),
) -> None:
    """Start only the dashboard (Gradio)."""
    import gradio as gr

    from .config import get_settings
    from .dashboard import DASHBOARD_CSS, build_dashboard

    build_dashboard().launch(
        server_name=host,
        server_port=port,
        css=DASHBOARD_CSS,
        theme=gr.themes.Soft(),
        max_file_size=f"{get_settings().max_upload_mb}mb",
    )


if __name__ == "__main__":
    app()
