"""Self-contained HTML data-quality report (charts embedded as PNG, no external assets, works offline)."""

import datetime as dt
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from jinja2 import Environment, select_autoescape

from . import charts

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
:root{--fg:#0f172a;--muted:#64748b;--bg:#ffffff;--card:#f8fafc;--line:#e2e8f0;--accent:#2563eb;
--crit:#b91c1c;--warn:#b45309;--info:#475569}
@media (prefers-color-scheme: dark){:root{--fg:#e2e8f0;--muted:#94a3b8;--bg:#0b1220;--card:#111a2e;--line:#1f2a44}}
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg);
margin:0;padding:24px;max-width:1150px;margin:auto}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 10px;border-bottom:1px solid var(--line);
padding-bottom:4px} .muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.kpi{font-size:22px;font-weight:600} table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--card);position:sticky;top:0} .sev-critical{color:var(--crit);font-weight:600}
.sev-warning{color:var(--warn);font-weight:600} .sev-info{color:var(--info)}
img{max-width:100%;height:auto;background:#fff;border-radius:8px}
details{margin:6px 0} summary{cursor:pointer;font-weight:600} code{font-size:12px}
.scroll{overflow-x:auto}
</style></head><body>
<h1>{{ title }}</h1>
<div class="muted">Generated {{ generated }} &middot; featureforge {{ version }}{% if source %} &middot; source: {{ source }}{% endif %}</div>

<h2>Summary</h2>
<div class="grid">
 <div class="card"><div class="muted">Quality score</div><div class="kpi">{{ p.quality_score }}/100</div></div>
 <div class="card"><div class="muted">Rows</div><div class="kpi">{{ "{:,}".format(p.dataset.rows) }}</div></div>
 <div class="card"><div class="muted">Columns</div><div class="kpi">{{ p.dataset.columns }}</div></div>
 <div class="card"><div class="muted">Missing cells</div><div class="kpi">{{ pct(p.dataset.missing_pct) }}</div></div>
 <div class="card"><div class="muted">Duplicate rows</div><div class="kpi">{{ "{:,}".format(p.dataset.duplicate_rows) }}</div></div>
 <div class="card"><div class="muted">Warnings</div><div class="kpi">{{ p.warnings|length }}</div></div>
</div>
{% if p.dataset.sampled %}<p class="muted">Column statistics computed on a random sample of {{ "{:,}".format(p.dataset.sample_rows) }} rows; duplicates and missingness use all rows.</p>{% endif %}
<p class="muted">Quality score = 100 &times; (1 &minus; 0.5&middot;missing share &minus; 0.3&middot;duplicate share &minus; 0.2&middot;share of columns with a warning).</p>

<h2>Warnings</h2>
{% if p.warnings %}<div class="scroll"><table><tr><th>Severity</th><th>Check</th><th>Column</th><th>Detail</th></tr>
{% for w in warnings_sorted %}<tr><td class="sev-{{ w.severity }}">{{ w.severity }}</td><td>{{ w.code }}</td>
<td>{{ w.column or "" }}</td><td>{{ w.message }}</td></tr>{% endfor %}</table></div>
{% else %}<p>No issues found.</p>{% endif %}

<h2>Charts</h2>
{% for c in top_charts %}{% if c %}<p><img alt="chart" src="data:image/png;base64,{{ c }}"></p>{% endif %}{% endfor %}

<h2>Columns</h2>
<div class="scroll"><table><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Details</th></tr>
{% for c in p.columns %}<tr><td><code>{{ c.name }}</code></td><td>{{ c.type }}</td><td>{{ pct(c.missing_pct) }}</td>
<td>{{ c.unique }}</td><td>{% if c.stats %}mean {{ fmt(c.stats.mean) }}, median {{ fmt(c.stats.median) }}, min {{ fmt(c.stats.min) }}, max {{ fmt(c.stats.max) }}, skew {{ fmt(c.stats.skew) }}{% elif c.min %}{{ c.min }} &rarr; {{ c.max }}{% else %}{% for v in c.top_values[:3] %}{{ v.value }} ({{ v.count }}){% if not loop.last %}, {% endif %}{% endfor %}{% endif %}</td></tr>{% endfor %}
</table></div>

<h2>Column details</h2>
{% for c in p.columns %}{% set img = col_charts.get(c.name) %}{% if img %}<details><summary>{{ c.name }} ({{ c.type }})</summary>
<p><img alt="{{ c.name }}" src="data:image/png;base64,{{ img }}"></p></details>{% endif %}{% endfor %}

{% if p.target %}<h2>Target</h2>
<p>Target <code>{{ p.target.name }}</code>, task <b>{{ p.target.task }}</b>{% if p.target.classes %}; classes: {% for k, v in p.target.classes.items() %}{{ k }} {{ pct(v) }}{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}.</p>
{% endif %}

{% if steps %}<h2>Pipeline</h2>
<div class="scroll"><table><tr><th>#</th><th>Step</th><th>Rows</th><th>Columns</th><th>Added / removed</th><th>Notes</th></tr>
{% for s in steps %}<tr><td>{{ s.index }}</td><td><code>{{ s.step }}</code></td><td>{{ s.rows_before }} &rarr; {{ s.rows_after }}</td>
<td>{{ s.cols_before }} &rarr; {{ s.cols_after }}</td><td>+{{ s.added|length }} / &minus;{{ s.removed|length }}</td>
<td>{{ s.notes|join("; ") }}</td></tr>{% endfor %}</table></div>
{% if shape_chart %}<p><img alt="pipeline" src="data:image/png;base64,{{ shape_chart }}"></p>{% endif %}
{% endif %}

{% if drift %}<h2>Drift against reference</h2>
<p>Missing columns: {{ drift.missing_columns or "none" }}. New columns: {{ drift.new_columns or "none" }}.</p>
<div class="scroll"><table><tr><th>Column</th><th>Type</th><th>PSI</th><th>Status</th><th>KS stat</th></tr>
{% for d in drift.columns %}<tr><td>{{ d.column }}</td><td>{{ d.type }}</td><td>{{ d.psi }}</td><td>{{ d.status }}</td><td>{{ d.ks_stat if d.ks_stat is defined else "" }}</td></tr>{% endfor %}
</table></div>{% endif %}

{% if preview %}<h2>Output preview</h2><div class="scroll">{{ preview|safe }}</div>{% endif %}
</body></html>
"""

_SEV = {"critical": 0, "warning": 1, "info": 2}


def _pct(v: Optional[float]) -> str:
    return "" if v is None else f"{100 * v:.1f}%"


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)


def render_report(
    p: Dict[str, Any],
    title: str = "Data quality report",
    steps: Optional[List[Dict[str, Any]]] = None,
    drift: Optional[Dict[str, Any]] = None,
    preview: Optional[pd.DataFrame] = None,
    source: Optional[str] = None,
    max_column_charts: int = 40,
) -> str:
    from . import __version__

    env = Environment(autoescape=select_autoescape(default=True))
    tpl = env.from_string(TEMPLATE)
    col_charts: Dict[str, str] = {}
    for c in p["columns"][:max_column_charts]:
        fig = charts.histogram(c) if c.get("stats") else charts.top_values(c)
        img = charts.to_base64(fig)
        if img:
            col_charts[c["name"]] = img
    preview_html = None
    if preview is not None:
        preview_html = preview.head(20).to_html(index=False, border=0, escape=True, na_rep="")
    return tpl.render(
        title=title,
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=__version__,
        source=html.escape(source) if source else None,
        p=p,
        warnings_sorted=sorted(p["warnings"], key=lambda w: _SEV.get(w["severity"], 3)),
        top_charts=[
            charts.to_base64(charts.type_counts(p)),
            charts.to_base64(charts.missingness(p)),
            charts.to_base64(charts.correlation(p)),
            charts.to_base64(charts.target_association(p)),
        ],
        col_charts=col_charts,
        steps=steps,
        shape_chart=charts.to_base64(charts.pipeline_shape(steps or [])),
        drift=drift,
        preview=preview_html,
        pct=_pct,
        fmt=_fmt,
    )


def write_report(path: Union[str, Path], p: Dict[str, Any], **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(p, **kwargs), encoding="utf-8")
    return path
