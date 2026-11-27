"""End-to-end job runner shared by the CLI, the REST API and the dashboard.

``run_job`` = profile -> (auto) recipe -> optional leakage-safe split -> fit on train -> transform test -> write
artifacts. Output folder layout::

    <run>/
      data.csv / data.parquet        engineered dataset (train part when split)
      test.csv / test.parquet        engineered test part (only when split)
      pipeline/                      recipe.yaml, pipeline.joblib, metadata.json
      report.html                    data-quality report (source profile + pipeline steps + output preview)
      datacard.md                    dataset card
      profile.json, summary.json     machine-readable profile and run summary
      bundle.zip                     everything above
"""

import datetime as dt
import json
import logging
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .auto import auto_recipe
from .config import get_settings
from .datacard import build_datacard
from .io import sha256_file, write_table
from .pipeline import Pipeline
from .profile import drift_report, profile
from .recipe import Recipe
from .report import write_report
from .split import leakage_safe_split

logger = logging.getLogger(__name__)
RUN_ID = re.compile(r"^[a-f0-9]{12}$")


@dataclass
class RunResult:
    run_id: str
    directory: Path
    summary: Dict[str, Any]
    output: pd.DataFrame
    test_output: Optional[pd.DataFrame] = None
    files: Dict[str, str] = field(default_factory=dict)


class RunStore:
    """Run folders under ``<work_dir>/runs/<12 hex chars>``; ids are validated (no path traversal)."""

    def __init__(self, work_dir: Optional[Union[str, Path]] = None) -> None:
        self.root = Path(work_dir or get_settings().work_dir) / "runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def new(self) -> tuple:
        run_id = uuid.uuid4().hex[:12]
        d = self.root / run_id
        d.mkdir(parents=True)
        return run_id, d

    def path(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id or ""):
            raise KeyError("invalid run id")
        d = self.root / run_id
        if not d.is_dir():
            raise KeyError(f"run {run_id} not found")
        return d

    def file(self, run_id: str, name: str) -> Path:
        d = self.path(run_id)
        p = (d / name).resolve()
        if d.resolve() not in p.parents or not p.is_file():
            raise KeyError(f"file {name!r} not found in run {run_id}")
        return p

    def list(self) -> List[Dict[str, Any]]:
        out = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            s = d / "summary.json"
            if s.exists():
                out.append(json.loads(s.read_text(encoding="utf-8")))
        return out

    def cleanup(self, ttl_hours: Optional[int] = None) -> int:
        ttl = (ttl_hours if ttl_hours is not None else get_settings().run_ttl_hours) * 3600
        removed = 0
        for d in self.root.iterdir():
            if d.is_dir() and time.time() - d.stat().st_mtime > ttl:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return removed


def run_job(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    recipe: Optional[Recipe] = None,
    target: Optional[str] = None,
    level: str = "standard",
    test_size: Optional[float] = None,
    split_strategy: str = "auto",
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    source_name: str = "upload",
    source_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    leak_policy: str = "auto",
) -> RunResult:
    """Run the full workflow and write all artifacts to ``out_dir``."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if recipe is not None and target is None:
        target = recipe.target
    src_profile = profile(df, target=target)
    if recipe is None:
        recipe = auto_recipe(
            df, target=target, level=level, report=src_profile, name=f"auto-{level}", leak_policy=leak_policy
        )
    elif target is not None and recipe.target is None:
        recipe = recipe.model_copy(update={"target": target})

    split_report = None
    train, test = df, None
    if test_size:
        train, test, split_report = leakage_safe_split(
            df,
            target=recipe.target,
            test_size=test_size,
            strategy=split_strategy,
            group_column=group_column,
            time_column=time_column,
            random_state=recipe.random_state,
        )
    pipe = Pipeline(recipe)
    output = pipe.fit_transform(train)
    test_output = pipe.transform(test) if test is not None else None
    drift = drift_report(train, test) if test is not None else None

    pipe_dir = pipe.save(out_dir / "pipeline")
    files: Dict[str, str] = {}
    write_table(output, out_dir / "data.csv")
    write_table(output, out_dir / "data.parquet")
    files.update({"data.csv": "data.csv", "data.parquet": "data.parquet"})
    if test_output is not None:
        write_table(test_output, out_dir / "test.csv")
        write_table(test_output, out_dir / "test.parquet")
        files.update({"test.csv": "test.csv", "test.parquet": "test.parquet"})
    summary_pipe = pipe.summary()
    policies = [s.params.get("leak_policy") for s in recipe.steps if s.step == "drop_columns" and s.enabled]
    used_policy = next((x for x in policies if x), None)
    write_report(
        out_dir / "report.html",
        src_profile,
        title=f"Data quality report: {source_name}",
        steps=summary_pipe["steps"],
        drift=drift,
        preview=output,
        source=source_name,
        leaks=summary_pipe["leaks"] if used_policy else None,
        leak_policy=used_policy,
    )
    (out_dir / "profile.json").write_text(json.dumps(src_profile, indent=1, default=str), encoding="utf-8")
    hashes = {name: sha256_file(out_dir / name) for name in files}
    card = build_datacard(
        name=source_name,
        source_profile=src_profile,
        output=output,
        pipeline_summary=summary_pipe,
        recipe_yaml=recipe.to_yaml(),
        source_file=source_name,
        source_sha256=sha256_file(source_path) if source_path else None,
        output_sha256=hashes,
    )
    (out_dir / "datacard.md").write_text(card, encoding="utf-8")
    files.update(
        {
            "report.html": "report.html",
            "datacard.md": "datacard.md",
            "profile.json": "profile.json",
            "recipe.yaml": "pipeline/recipe.yaml",
            "pipeline.joblib": "pipeline/pipeline.joblib",
            "metadata.json": "pipeline/metadata.json",
        }
    )
    summary = {
        "run_id": run_id or out_dir.name,
        "source": source_name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - t0, 3),
        "input_shape": list(df.shape),
        "output_shape": list(output.shape),
        "test_output_shape": list(test_output.shape) if test_output is not None else None,
        "quality_score": src_profile["quality_score"],
        "warnings": len(src_profile["warnings"]),
        "target": pipe.ctx_.target,
        "task": pipe.ctx_.task,
        "split": split_report,
        "drifted_columns": drift["drifted"] if drift else None,
        "leak_policy": used_policy,
        "leaks": summary_pipe["leaks"],
        "pipeline": summary_pipe,
        "transform_warnings": pipe.warnings_,
        "files": files,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    files["summary.json"] = "summary.json"
    with zipfile.ZipFile(out_dir / "bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(set(files.values())):
            z.write(out_dir / rel, rel)
    files["bundle.zip"] = "bundle.zip"
    summary["files"] = files
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    logger.info("run %s finished in %.1fs: %s -> %s", summary["run_id"], summary["seconds"], df.shape, output.shape)
    _ = pipe_dir
    return RunResult(summary["run_id"], out_dir, summary, output, test_output, files)
