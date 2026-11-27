"""Matplotlib charts built from a profile dict (used by the HTML report and the dashboard)."""

import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ACCENT = "#2563eb"
MUTED = "#94a3b8"
WARN = "#d97706"


def _fig(w: float = 7.0, h: float = 3.2) -> Figure:
    fig = plt.figure(figsize=(w, h), dpi=110)
    return fig


def missingness(p: Dict[str, Any], top: int = 25) -> Optional[Figure]:
    cols = sorted(p["columns"], key=lambda c: -(c["missing_pct"] or 0))[:top]
    cols = [c for c in cols if c["missing"]]
    if not cols:
        return None
    fig = _fig(7, max(2.2, 0.28 * len(cols) + 0.8))
    ax = fig.add_subplot(111)
    ax.barh([c["name"] for c in cols][::-1], [100 * (c["missing_pct"] or 0) for c in cols][::-1], color=WARN)
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def type_counts(p: Dict[str, Any]) -> Figure:
    counts = p["dataset"]["type_counts"]
    fig = _fig(5, 2.6)
    ax = fig.add_subplot(111)
    names = list(counts)
    ax.bar(names, [counts[n] for n in names], color=ACCENT)
    ax.set_title("Inferred column types")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def histogram(col: Dict[str, Any]) -> Optional[Figure]:
    h = (col.get("stats") or {}).get("histogram")
    if not h or not h["counts"]:
        return None
    fig = _fig(4.2, 2.4)
    ax = fig.add_subplot(111)
    edges = h["edges"]
    widths = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    ax.bar(edges[:-1], h["counts"], width=widths, align="edge", color=ACCENT, edgecolor="white", linewidth=0.4)
    ax.set_title(col["name"], fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def top_values(col: Dict[str, Any], top: int = 10) -> Optional[Figure]:
    vals = col.get("top_values") or []
    if not vals:
        return None
    vals = vals[:top]
    fig = _fig(4.2, max(2.0, 0.25 * len(vals) + 0.8))
    ax = fig.add_subplot(111)
    ax.barh([str(v["value"])[:30] for v in vals][::-1], [v["count"] for v in vals][::-1], color=ACCENT)
    ax.set_title(col["name"], fontsize=10)
    fig.tight_layout()
    return fig


def correlation(p: Dict[str, Any]) -> Optional[Figure]:
    corr = p.get("correlations") or {}
    cols: List[str] = corr.get("columns") or []
    if len(cols) < 2:
        return None
    n = len(cols)
    fig = _fig(min(9, 2 + 0.38 * n), min(8, 1.6 + 0.35 * n))
    ax = fig.add_subplot(111)
    m = [[v if v is not None else 0 for v in row] for row in corr["matrix"]]
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), [c[:18] for c in cols], rotation=90, fontsize=7)
    ax.set_yticks(range(n), [c[:18] for c in cols], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Pearson correlation (numeric columns)")
    fig.tight_layout()
    return fig


def target_association(p: Dict[str, Any], top: int = 20) -> Optional[Figure]:
    t = p.get("target")
    if not t or not t.get("associations"):
        return None
    a = t["associations"][:top]
    fig = _fig(7, max(2.2, 0.28 * len(a) + 0.8))
    ax = fig.add_subplot(111)
    ax.barh([x["column"][:30] for x in a][::-1], [x["nmi"] for x in a][::-1], color=ACCENT)
    ax.set_xlabel("normalised mutual information with target")
    ax.set_title(f"Association with target '{t['name']}'")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def pipeline_shape(steps: List[Dict[str, Any]]) -> Optional[Figure]:
    if not steps:
        return None
    fig = _fig(7, 2.8)
    ax = fig.add_subplot(111)
    xs = ["input"] + [f"{s['index']}. {s['step']}" for s in steps]
    cols = [steps[0]["cols_before"]] + [s["cols_after"] for s in steps]
    rows = [steps[0]["rows_before"]] + [s["rows_after"] for s in steps]
    ax.plot(xs, cols, marker="o", color=ACCENT, label="columns")
    ax2 = ax.twinx()
    ax2.plot(xs, rows, marker="s", color=MUTED, label="rows")
    ax.set_ylabel("columns")
    ax2.set_ylabel("rows")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.set_title("Shape after each step")
    fig.tight_layout()
    return fig


def to_base64(fig: Optional[Figure]) -> Optional[str]:
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
