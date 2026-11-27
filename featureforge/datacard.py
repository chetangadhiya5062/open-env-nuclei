"""Dataset card (Markdown) following *Datasheets for Datasets* (Gebru et al., 2021).

Facts the software knows (composition, preprocessing, quality findings, checksums) are filled in. Questions only the
data owner can answer (motivation, collection, consent, intended use) are left as clearly marked prompts: the card
never invents them.
"""

import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd

TODO = "_To be completed by the data owner._"


def _table(rows: List[List[Any]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return "\n".join(out)


def build_datacard(
    name: str,
    source_profile: Dict[str, Any],
    output: pd.DataFrame,
    pipeline_summary: Optional[Dict[str, Any]] = None,
    recipe_yaml: Optional[str] = None,
    source_file: Optional[str] = None,
    source_sha256: Optional[str] = None,
    output_sha256: Optional[Dict[str, str]] = None,
) -> str:
    from . import __version__

    ds = source_profile["dataset"]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Dataset card: {name}",
        "",
        f"Generated {now} by featureforge {__version__}. Sections marked _To be completed_ need the data owner.",
        "",
        "## Motivation",
        "- Why was the dataset created, by whom, and who funded it? " + TODO,
        "",
        "## Composition",
        f"- Source file: `{source_file or 'n/a'}`" + (f" (sha256 `{source_sha256}`)" if source_sha256 else ""),
        f"- Source: {ds['rows']:,} rows x {ds['columns']} columns; {ds['duplicate_rows']:,} exact duplicate rows; "
        f"{100 * (ds['missing_pct'] or 0):.1f}% missing cells.",
        f"- Output: {len(output):,} rows x {output.shape[1]} columns.",
        f"- Inferred column types (source): {ds['type_counts']}.",
        "- What does one row represent? Does the data contain personal or sensitive information? " + TODO,
        "",
        "### Source columns",
        _table(
            [
                [f"`{c['name']}`", c["type"], f"{100 * (c['missing_pct'] or 0):.1f}%", c["unique"]]
                for c in source_profile["columns"]
            ],
            ["column", "type", "missing", "unique"],
        ),
        "",
    ]
    tgt = source_profile.get("target")
    if tgt:
        lines += [f"- Target: `{tgt['name']}` ({tgt['task']})."]
        if tgt.get("classes"):
            lines += ["- Class balance: " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in tgt["classes"].items()) + "."]
        lines.append("")
    lines += [
        "## Collection process",
        "- How, when and from whom was the data collected? Was consent obtained? " + TODO,
        "",
    ]
    lines += ["## Preprocessing / cleaning / labelling"]
    if pipeline_summary:
        lines.append(
            f"Applied recipe `{pipeline_summary['recipe']}` (task {pipeline_summary['task']}, fitted on "
            f"{pipeline_summary['fit_rows']:,} rows at {pipeline_summary['fitted_at']}):"
        )
        lines.append("")
        lines.append(
            _table(
                [
                    [
                        s["index"],
                        f"`{s['step']}`",
                        f"{s['rows_before']} -> {s['rows_after']}",
                        f"{s['cols_before']} -> {s['cols_after']}",
                        "; ".join(s["notes"])[:200],
                    ]
                    for s in pipeline_summary["steps"]
                ],
                ["#", "step", "rows", "columns", "notes"],
            )
        )
        lines.append("")
        leaks = pipeline_summary.get("leaks") or []
        if leaks:
            lines.append("")
            lines.append("Target-leak check (columns that copy or determine the target):")
            lines.append("")
            lines.append(
                _table(
                    [[f"`{x['column']}`", x["code"], x["confidence"], x["nmi"], x["action"]] for x in leaks],
                    ["column", "finding", "confidence", "normalised MI", "action"],
                )
            )
            lines.append("")
        lines.append("All statistics (imputation values, encodings, bounds, selected features) were learned on the")
        lines.append("training data only and are stored in `pipeline.joblib`; `recipe.yaml` reproduces the run.")
    else:
        lines.append("No preprocessing applied.")
    if recipe_yaml:
        lines += [
            "",
            "<details><summary>recipe.yaml</summary>",
            "",
            "```yaml",
            recipe_yaml.strip(),
            "```",
            "",
            "</details>",
        ]
    lines += ["", "## Data quality findings (source)", f"Quality score: {source_profile['quality_score']}/100.", ""]
    ws = [w for w in source_profile["warnings"] if w["severity"] in ("critical", "warning")]
    if ws:
        lines.append(
            _table(
                [[w["severity"], w["code"], w["column"] or "", w["message"]] for w in ws[:60]],
                ["severity", "check", "column", "detail"],
            )
        )
    else:
        lines.append("No warnings.")
    lines += [
        "",
        "## Uses",
        "- Intended uses and uses to avoid: " + TODO,
        "- Known limitations: imputed values are estimates (see `__was_missing` indicator columns); outlier capping "
        "changes extreme values; automatic type inference can be wrong for codes stored as numbers.",
        "",
        "## Distribution and maintenance",
        "- License, access, who maintains the dataset, how errors are reported: " + TODO,
    ]
    if output_sha256:
        lines += [
            "",
            "## Checksums",
            _table([[f"`{k}`", f"`{v}`"] for k, v in output_sha256.items()], ["file", "sha256"]),
        ]
    return "\n".join(lines) + "\n"
