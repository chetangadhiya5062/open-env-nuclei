"""Ground-truth grader: final score in [0, 1].

The grader is **separate from the reward** and is only revealed when the episode ends. It compares the
agent's table with the hidden clean table, aligning rows on ``customer_id``.

    score = 0.6 * cell_accuracy + 0.2 * row_fidelity + 0.2 * schema_score

* cell_accuracy - mean over *all* ground-truth cells. A lost row contributes zeros.
    - categorical / text / date: 1 if exactly equal, else 0
    - numeric: 1 if equal, otherwise partial credit ``max(0, 1 - |pred - true| / std(true column))``.
      Imputed values can almost never be exact, so a close estimate earns most of the credit;
      a string or a null earns 0.
* row_fidelity  - ``matched_true_rows / max(rows_in_agent_table, rows_in_truth)``: penalises both lost
  rows and leftover duplicates.
* schema_score  - fraction of columns whose type/format is valid (see ``quality.column_issues``, which
  must report no type, format or allowed-value problem and no nulls).

A table identical to the ground truth scores exactly 1.0.
"""

from typing import Dict

import numpy as np
import pandas as pd

from ..schema import ID_COLUMN, SCHEMA
from .quality import column_issues, is_number

W_CELLS, W_ROWS, W_SCHEMA = 0.6, 0.2, 0.2


def improvement(score: float, do_nothing_score: float) -> float:
    """Share of the possible improvement achieved: 0 = no better than leaving the data alone, 1 = perfect.

    ``(score - do_nothing) / (1 - do_nothing)``. Negative when the agent made the table worse than the untouched
    one. If the untouched table is already perfect there is nothing to improve and the result is 0.0.
    """
    room = 1.0 - do_nothing_score
    if room < 1e-9:
        return 0.0
    return (score - do_nothing_score) / room


def _cell_score(pred, true, scale: float, numeric: bool) -> float:
    if pd.isna(pred):
        return 0.0
    if numeric:
        if not is_number(pred):
            return 0.0
        diff = abs(float(pred) - float(true))
        return 1.0 if diff < 1e-9 else max(0.0, 1.0 - diff / scale)
    return 1.0 if pred == true else 0.0


def grade(df: pd.DataFrame, truth: pd.DataFrame) -> Dict[str, float]:
    n_truth = len(truth)
    ids = pd.to_numeric(df[ID_COLUMN], errors="coerce") if ID_COLUMN in df.columns else pd.Series(dtype=float)
    first_row_for_id = {}
    for pos, rid in enumerate(ids.tolist()):
        if not np.isnan(rid):
            first_row_for_id.setdefault(int(rid), pos)

    scales = {c.name: max(float(truth[c.name].std(ddof=0)), 1e-9) for c in SCHEMA if c.is_numeric}
    total = 0.0
    matched = 0
    for _, true_row in truth.iterrows():
        pos = first_row_for_id.get(int(true_row[ID_COLUMN]))
        if pos is None:
            continue
        matched += 1
        pred_row = df.iloc[pos]
        for spec in SCHEMA:
            if spec.name not in df.columns:
                continue
            total += _cell_score(pred_row[spec.name], true_row[spec.name], scales.get(spec.name, 1.0), spec.is_numeric)
    cell_accuracy = total / (n_truth * len(SCHEMA))
    row_fidelity = matched / max(len(df), n_truth)
    schema_score = float(
        np.mean([1.0 if spec.name in df.columns and not column_issues(df[spec.name], spec) else 0.0 for spec in SCHEMA])
    )
    score = W_CELLS * cell_accuracy + W_ROWS * row_fidelity + W_SCHEMA * schema_score
    return {
        "score": round(float(min(1.0, max(0.0, score))), 6),
        "cell_accuracy": round(float(cell_accuracy), 6),
        "row_fidelity": round(float(row_fidelity), 6),
        "schema_score": round(schema_score, 6),
    }
