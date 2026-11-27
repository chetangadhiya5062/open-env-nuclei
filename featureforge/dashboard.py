"""FeatureForge dashboard (Gradio): upload -> profile -> build pipeline -> run -> export -> apply to new data.

Mounted at ``/ui`` by ``featureforge.api`` (``fe serve``) or run alone with ``fe dashboard``. The handler functions
are plain Python (``load_file``, ``do_profile``, ``make_recipe``, ``do_run``, ``apply_new``) so they are unit-tested
without a browser.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import __version__, charts
from .auto import auto_recipe
from .config import get_settings
from .io import SUPPORTED_READ, TableReadError, read_table, sha256_file, write_table
from .pipeline import Pipeline, PipelineError
from .profile import drift_report, profile
from .recipe import Recipe
from .report import write_report
from .samples import TARGETS, make_dataset
from .service import RunStore, run_job

logger = logging.getLogger(__name__)
NONE = "(none)"
SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}

CSS = """
.ff-kpi {font-size: 1.05rem}
.ff-header h1 {margin-bottom: 0}
footer {visibility: hidden}
"""

INTRO = f"""<div class="ff-header">

# FeatureForge
Clean, profile and engineer features for tabular data, reproducibly. v{__version__}

</div>

**1 Data** upload a file (CSV, TSV, Excel, Parquet, JSON) or load a sample &rarr; **2 Profile** quality report &rarr;
**3 Pipeline** generate or edit a recipe and run it &rarr; **4 Export** download everything &rarr;
**5 Apply** reuse the fitted pipeline on new data. Or press **One-click auto** on the first tab.
"""


def _empty_state() -> Dict[str, Any]:
    return {"df": None, "name": None, "path": None, "profile": None, "run_dir": None, "summary": None}


def _err(msg: str) -> str:
    return f"**Error:** {msg}"


# ------------------------------------------------------------------------------------------------- 1 data
def load_file(path: Optional[str], state: Dict[str, Any]) -> Tuple[Dict[str, Any], str, pd.DataFrame, List[str]]:
    state = _empty_state()
    if not path:
        return state, _err("choose a file first"), pd.DataFrame(), [NONE]
    p = Path(path)
    if p.stat().st_size > get_settings().max_upload_bytes:
        return state, _err(f"file is larger than {get_settings().max_upload_mb} MB"), pd.DataFrame(), [NONE]
    try:
        df = read_table(p)
    except TableReadError as exc:
        return state, _err(str(exc)), pd.DataFrame(), [NONE]
    state.update(df=df, name=p.name, path=str(p))
    return state, _loaded_msg(df, p.name), df.head(get_settings().preview_rows), [NONE] + list(df.columns)


def load_sample(kind: str, rows: int, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str, pd.DataFrame, Any]:
    df = make_dataset(kind, int(rows), seed=0)
    folder = RunStore().root.parent / "samples"
    path = write_table(df, folder / f"sample_{kind}.csv")
    state = _empty_state()
    state.update(df=df, name=path.name, path=str(path))
    return state, _loaded_msg(df, path.name), df.head(get_settings().preview_rows), TARGETS[kind]


def _loaded_msg(df: pd.DataFrame, name: str) -> str:
    return f"Loaded **{name}**: {len(df):,} rows x {df.shape[1]} columns."


def target_choices(state: Dict[str, Any]) -> List[str]:
    df = state.get("df")
    return [NONE] + (list(df.columns) if df is not None else [])


def _target(t: Optional[str]) -> Optional[str]:
    return None if t in (None, "", NONE) else t


# ---------------------------------------------------------------------------------------------- 2 profile
def do_profile(state: Dict[str, Any], target: Optional[str]):
    df = state.get("df")
    if df is None:
        return (state, _err("load data first"), pd.DataFrame(), None, None, None, None, None, [])
    try:
        p = profile(df, target=_target(target))
    except KeyError as exc:
        return (state, _err(str(exc)), pd.DataFrame(), None, None, None, None, None, [])
    state["profile"] = p
    ds = p["dataset"]
    crit = sum(w["severity"] == "critical" for w in p["warnings"])
    kpis = (
        f"### Quality score {p['quality_score']}/100\n"
        f"{ds['rows']:,} rows &middot; {ds['columns']} columns &middot; "
        f"{100 * (ds['missing_pct'] or 0):.1f}% missing cells &middot; {ds['duplicate_rows']:,} duplicate rows &middot; "
        f"{len(p['warnings'])} findings ({crit} critical)"
        + (f"\n\n_Statistics on a {ds['sample_rows']:,}-row sample._" if ds["sampled"] else "")
    )
    if p.get("target"):
        kpis += f"\n\nTarget **{p['target']['name']}** &rarr; task **{p['target']['task']}**."
    warns = pd.DataFrame(sorted(p["warnings"], key=lambda w: SEV_ORDER.get(w["severity"], 3)))
    if warns.empty:
        warns = pd.DataFrame(columns=["severity", "code", "column", "message"])
    else:
        warns = warns[["severity", "code", "column", "message"]]
    folder = Path(RunStore().root.parent / "reports")
    folder.mkdir(parents=True, exist_ok=True)
    rpt = write_report(
        folder / f"{Path(state['name']).stem}_report.html", p, title=f"Data quality report: {state['name']}"
    )
    return (
        state,
        kpis,
        warns,
        charts.type_counts(p),
        charts.missingness(p),
        charts.correlation(p),
        charts.target_association(p),
        str(rpt),
        [c["name"] for c in p["columns"]],
    )


def column_detail(state: Dict[str, Any], column: Optional[str]):
    p = state.get("profile")
    if not p or not column:
        return None, ""
    col = next((c for c in p["columns"] if c["name"] == column), None)
    if col is None:
        return None, ""
    fig = charts.histogram(col) if col.get("stats") else charts.top_values(col)
    shown = {k: v for k, v in col.items() if k not in ("top_values",)}
    if "stats" in shown and shown["stats"]:
        shown["stats"] = {k: v for k, v in shown["stats"].items() if k != "histogram"}
    return fig, "```json\n" + json.dumps(shown, indent=1, default=str) + "\n```"


# --------------------------------------------------------------------------------------------- 3 pipeline
def make_recipe(state: Dict[str, Any], target: Optional[str], level: str, leak_policy: str = "auto"):
    df = state.get("df")
    if df is None:
        return _err("load data first"), "", []
    try:
        report = state.get("profile")
        if report and (report.get("target") or {}).get("name") != _target(target):
            report = None  # profiled with another target: leak checks must be redone
        r = auto_recipe(
            df, target=_target(target), level=level, report=report, name=f"auto-{level}", leak_policy=leak_policy
        )
    except KeyError as exc:
        return _err(str(exc)), "", []
    names = [f"{i + 1}. {s.step}" for i, s in enumerate(r.steps)]
    return (
        f"Generated **{level}** recipe with {len(r.steps)} steps. Edit the YAML or untick steps, then run.",
        r.to_yaml(),
        names,
    )


def step_names(recipe_yaml: str) -> List[str]:
    try:
        r = Recipe.from_text(recipe_yaml)
    except Exception:
        return []
    return [f"{i + 1}. {s.step}" for i, s in enumerate(r.steps)]


def _apply_enabled(recipe_yaml: str, enabled: Optional[List[str]]) -> Recipe:
    r = Recipe.from_text(recipe_yaml)
    if enabled is not None:
        keep = {int(e.split(".")[0]) - 1 for e in enabled}
        for i, s in enumerate(r.steps):
            s.enabled = s.enabled and i in keep
    return r


def validate_yaml(recipe_yaml: str) -> str:
    try:
        r = Recipe.from_text(recipe_yaml)
    except Exception as exc:
        return _err(f"invalid recipe: {exc}")
    errors = r.validate_steps()
    return _err("; ".join(errors)) if errors else f"Recipe is valid ({len(r.steps)} steps)."


def load_recipe_file(path: Optional[str]) -> Tuple[str, List[str]]:
    if not path:
        return "", []
    text = Path(path).read_text(encoding="utf-8")
    return text, step_names(text)


def do_run(
    state: Dict[str, Any],
    recipe_yaml: str,
    enabled: Optional[List[str]],
    target: Optional[str],
    test_size: float,
    strategy: str,
    group_col: Optional[str],
    time_col: Optional[str],
):
    df = state.get("df")
    blank = (state, None, pd.DataFrame(), None, pd.DataFrame(), [], None)
    if df is None:
        return (blank[0], _err("load data first")) + blank[2:]
    if not (recipe_yaml or "").strip():
        return (blank[0], _err("generate or paste a recipe first")) + blank[2:]
    try:
        r = _apply_enabled(recipe_yaml, enabled)
        if _target(target) and not r.target:
            r = r.model_copy(update={"target": _target(target)})
        errors = r.validate_steps()
        if errors:
            raise ValueError("; ".join(errors))
        store = RunStore()
        run_id, d = store.new()
        res = run_job(
            df,
            out_dir=d,
            recipe=r,
            test_size=float(test_size) or None,
            split_strategy=strategy,
            group_column=_target(group_col),
            time_column=_target(time_col),
            source_name=state["name"] or "data",
            source_path=Path(state["path"]) if state.get("path") else None,
            run_id=run_id,
        )
    except (PipelineError, KeyError, ValueError) as exc:
        return (blank[0], _err(str(exc))) + blank[2:]
    state["run_dir"] = str(d)
    state["summary"] = res.summary
    s = res.summary
    msg = (
        f"Run **{run_id}** finished in {s['seconds']}s: {tuple(s['input_shape'])} &rarr; {tuple(s['output_shape'])}"
        f" (task {s['task']})."
    )
    if s.get("split"):
        sp = s["split"]
        msg += (
            f"\n\nLeakage-safe split ({sp['strategy']}): {sp['train_rows']} train / {sp['test_rows']} test rows; "
            f"{sp['shared_feature_rows']} feature rows shared between them. Pipeline fitted on train only."
        )
    for leak in s.get("leaks") or []:
        msg += (
            f"\n\n**Target leak:** `{leak['column']}` ({leak['code']}, {leak['confidence']} confidence, "
            f"normalised MI {leak['nmi']}) &rarr; {leak['action']}"
        )
    if s.get("drifted_columns"):
        msg += f"\n\nColumns with major train/test drift: {s['drifted_columns']}"
    steps = pd.DataFrame(
        [
            {
                "#": x["index"],
                "step": x["step"],
                "rows": f"{x['rows_before']} -> {x['rows_after']}",
                "columns": f"{x['cols_before']} -> {x['cols_after']}",
                "added": len(x["added"]),
                "removed": len(x["removed"]),
                "seconds": x["seconds"],
                "notes": "; ".join(x["notes"]),
            }
            for x in s["pipeline"]["steps"]
        ]
    )
    return (
        state,
        msg,
        steps,
        charts.pipeline_shape(s["pipeline"]["steps"]),
        res.output.head(get_settings().preview_rows),
        list(df.columns),
        None,
    )


def before_after(state: Dict[str, Any], column: Optional[str]) -> str:
    """Markdown comparison of one source column and the output columns derived from it."""
    df = state.get("df")
    if df is None or not state.get("run_dir") or not column:
        return ""
    out = pd.read_parquet(Path(state["run_dir"]) / "data.parquet")
    derived = [c for c in out.columns if c == column or c.startswith(f"{column}__") or c.startswith(f"{column}=")]
    lines = [f"**Before** `{column}` ({df[column].dtype}, {int(df[column].isna().sum())} missing):", ""]
    lines.append(_md_table(df[[column]].head(8)))
    if not derived:
        lines += [
            "",
            "_Not in the output (dropped, encoded under another name, or used only to build other features)._",
        ]
        return "\n".join(lines)
    lines += ["", f"**After**: {len(derived)} column(s)", ""]
    lines.append(_md_table(out[derived[:8]].head(8)))
    return "\n".join(lines)


def _md_table(df: pd.DataFrame) -> str:
    def cell(v: Any) -> str:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return ""
        return (f"{v:.4g}" if isinstance(v, float) else str(v)).replace("|", "\\|")[:40]

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * df.shape[1]
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


# ----------------------------------------------------------------------------------------------- 4 export
def export_files(state: Dict[str, Any]):
    d = state.get("run_dir")
    if not d:
        return [], _err("run a pipeline first")
    d = Path(d)
    files = state["summary"]["files"]
    order = [
        "data.csv",
        "data.parquet",
        "test.csv",
        "test.parquet",
        "bundle.zip",
        "recipe.yaml",
        "report.html",
        "datacard.md",
        "pipeline.joblib",
        "metadata.json",
        "summary.json",
    ]
    paths = [str(d / files[k]) for k in order if k in files]
    card = (d / "datacard.md").read_text(encoding="utf-8")
    return paths, card


# ------------------------------------------------------------------------------------------------ 5 apply
def apply_new(state: Dict[str, Any], path: Optional[str], fmt: str):
    d = state.get("run_dir")
    if not d:
        return None, _err("run a pipeline first"), pd.DataFrame()
    if not path:
        return None, _err("choose a file"), pd.DataFrame()
    try:
        new = read_table(path)
        pipe = Pipeline.load(Path(d) / "pipeline")
        out = pipe.transform(new)
    except (TableReadError, PipelineError) as exc:
        return None, _err(str(exc)), pd.DataFrame()
    target = Path(d) / "applied" / f"{Path(path).stem}_features.{fmt}"
    write_table(out, target)
    drift = drift_report(state["df"], new)
    msg = [f"Transformed {new.shape} &rarr; {out.shape} (sha256 `{sha256_file(target)[:16]}…`)."]
    msg += [f"- warning: {w}" for w in pipe.warnings_]
    if drift["drifted"]:
        msg.append(f"- major drift versus the training data in: {drift['drifted']}")
    elif drift["columns"]:
        msg.append("- no column with major drift versus the training data (PSI < 0.25)")
    return str(target), "\n".join(msg), out.head(get_settings().preview_rows)


# ------------------------------------------------------------------------------------------------- layout
def build_dashboard():
    import gradio as gr

    exts = sorted(SUPPORTED_READ)
    with gr.Blocks(title="FeatureForge") as demo:
        state = gr.State(_empty_state())
        gr.Markdown(INTRO)
        with gr.Tabs() as tabs:
            with gr.Tab("1 · Data", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        upload = gr.File(label=f"Upload ({', '.join(exts)})", file_types=exts, type="filepath")
                    with gr.Column(scale=1):
                        sample_kind = gr.Dropdown(["churn", "housing", "transactions"], value="churn", label="Sample")
                        sample_rows = gr.Slider(200, 20000, value=2000, step=100, label="Sample rows")
                        sample_btn = gr.Button("Load sample")
                status = gr.Markdown()
                with gr.Row():
                    target = gr.Dropdown([NONE], value=NONE, label="Target column (optional)", allow_custom_value=True)
                    auto_btn = gr.Button("One-click auto (profile + standard recipe + run)", variant="primary")
                preview = gr.Dataframe(label="Preview", interactive=False, wrap=True)
            with gr.Tab("2 · Profile", id=1):
                prof_btn = gr.Button("Profile data", variant="primary")
                kpis = gr.Markdown(elem_classes="ff-kpi")
                warns = gr.Dataframe(label="Findings", interactive=False, wrap=True)
                with gr.Row():
                    p_types = gr.Plot(label="Types")
                    p_missing = gr.Plot(label="Missingness")
                with gr.Row():
                    p_corr = gr.Plot(label="Correlation")
                    p_target = gr.Plot(label="Target association")
                with gr.Row():
                    col_pick = gr.Dropdown([], label="Column detail")
                    report_file = gr.File(label="HTML report", interactive=False)
                with gr.Row():
                    col_plot = gr.Plot()
                    col_json = gr.Markdown()
            with gr.Tab("3 · Pipeline", id=2):
                with gr.Row():
                    level = gr.Radio(["clean", "standard", "full"], value="standard", label="Auto level")
                    leak_pol = gr.Radio(
                        ["auto", "drop", "warn"],
                        value="auto",
                        label="Target-leak policy",
                        info="auto: drop high-confidence leaks, warn on the rest; drop: drop all flagged; warn: keep all",
                    )
                    gen_btn = gr.Button("Generate recipe from data")
                    recipe_upload = gr.File(
                        label="…or import recipe (.yaml/.json)", file_types=[".yaml", ".yml", ".json"]
                    )
                recipe_msg = gr.Markdown()
                enabled = gr.CheckboxGroup([], label="Steps to run (untick to skip)")
                recipe_code = gr.Code(label="Recipe (YAML, editable)", language="yaml", lines=18)
                with gr.Accordion("Leakage-safe train/test split (optional)", open=False):
                    with gr.Row():
                        test_size = gr.Slider(0, 0.5, value=0, step=0.05, label="Test share (0 = no split)")
                        strategy = gr.Dropdown(
                            ["auto", "random", "stratified", "group", "time"], value="auto", label="Strategy"
                        )
                        group_col = gr.Dropdown([NONE], value=NONE, label="Group column")
                        time_col = gr.Dropdown([NONE], value=NONE, label="Time column")
                with gr.Row():
                    val_btn = gr.Button("Validate recipe")
                    run_btn = gr.Button("Run pipeline", variant="primary")
                run_msg = gr.Markdown()
                steps_tbl = gr.Dataframe(label="Step report", interactive=False, wrap=True)
                shape_plot = gr.Plot(label="Shape after each step")
                with gr.Row():
                    ba_col = gr.Dropdown([], label="Before / after for column")
                ba_md = gr.Markdown()
                out_preview = gr.Dataframe(label="Output preview", interactive=False)
            with gr.Tab("4 · Export", id=3):
                exp_btn = gr.Button("Refresh downloads")
                downloads = gr.File(label="Artifacts", file_count="multiple", interactive=False)
                card_md = gr.Markdown()
            with gr.Tab("5 · Apply", id=4):
                gr.Markdown("Apply the pipeline fitted in tab 3 to new data with the same columns (target optional).")
                with gr.Row():
                    new_file = gr.File(label="New data", file_types=exts, type="filepath")
                    fmt = gr.Radio(["csv", "parquet", "json"], value="csv", label="Output format")
                apply_btn = gr.Button("Transform", variant="primary")
                apply_out = gr.File(label="Result", interactive=False)
                apply_msg = gr.Markdown()
                apply_preview = gr.Dataframe(interactive=False)

        def _set_cols(st):
            ch = target_choices(st)
            return (
                gr.update(choices=ch, value=NONE),
                gr.update(choices=ch, value=NONE),
                gr.update(choices=ch, value=NONE),
            )

        upload.change(lambda p, s: load_file(p, s)[:3], [upload, state], [state, status, preview]).then(
            _set_cols, state, [target, group_col, time_col]
        )
        sample_btn.click(
            lambda k, n, s: load_sample(k, n, s)[:3], [sample_kind, sample_rows, state], [state, status, preview]
        ).then(_set_cols, state, [target, group_col, time_col]).then(
            lambda k: gr.update(value=TARGETS[k]), sample_kind, target
        )

        prof_outputs = [state, kpis, warns, p_types, p_missing, p_corr, p_target, report_file, col_pick]

        def _prof(s, t):
            r = list(do_profile(s, t))
            r[-1] = gr.update(choices=r[-1], value=None)
            return r

        prof_btn.click(_prof, [state, target], prof_outputs)
        col_pick.change(column_detail, [state, col_pick], [col_plot, col_json])

        def _gen(s, t, lv, lp):
            m, y, names = make_recipe(s, t, lv, lp)
            return m, y, gr.update(choices=names, value=names)

        gen_btn.click(_gen, [state, target, level, leak_pol], [recipe_msg, recipe_code, enabled])

        def _imp(p):
            y, names = load_recipe_file(p)
            return y, gr.update(choices=names, value=names), validate_yaml(y) if y else ""

        recipe_upload.change(_imp, recipe_upload, [recipe_code, enabled, recipe_msg])
        val_btn.click(validate_yaml, recipe_code, recipe_msg)

        run_outputs = [state, run_msg, steps_tbl, shape_plot, out_preview, ba_col, ba_md]

        def _run(s, y, en, t, ts, stg, g, tc):
            r = list(do_run(s, y, en, t, ts, stg, g, tc))
            r[5] = gr.update(choices=r[5], value=None)
            r[6] = ""
            return r

        run_inputs = [state, recipe_code, enabled, target, test_size, strategy, group_col, time_col]
        run_btn.click(_run, run_inputs, run_outputs).then(export_files, state, [downloads, card_md])
        ba_col.change(before_after, [state, ba_col], ba_md)
        exp_btn.click(export_files, state, [downloads, card_md])
        apply_btn.click(apply_new, [state, new_file, fmt], [apply_out, apply_msg, apply_preview])

        def _auto(s, t, lp):
            pr = _prof(s, t)
            s2 = pr[0]
            m, y, names = make_recipe(s2, t, "standard", lp)
            r = list(do_run(s2, y, names, t, 0, "auto", None, None))
            r[5] = gr.update(choices=r[5], value=None)
            r[6] = ""
            files, card = export_files(r[0])
            return pr[1:] + [m, y, gr.update(choices=names, value=names)] + r + [files, card, gr.Tabs(selected=3)]

        auto_btn.click(
            _auto,
            [state, target, leak_pol],
            prof_outputs[1:] + [recipe_msg, recipe_code, enabled] + run_outputs + [downloads, card_md, tabs],
        )
    return demo


DASHBOARD_CSS = CSS
