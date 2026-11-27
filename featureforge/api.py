"""REST API (FastAPI). Run: ``fe serve`` or ``uvicorn featureforge.api:app``. Docs at ``/docs``; dashboard at ``/ui``.

| method | path | what |
|---|---|---|
| GET  | /health | liveness |
| GET  | /v1/info | version, limits, supported formats |
| GET  | /v1/steps | step catalog with parameter schemas |
| POST | /v1/profile | upload a file -> profile JSON |
| POST | /v1/recipes/auto | upload a file -> generated recipe |
| POST | /v1/recipes/validate | recipe JSON -> list of errors |
| POST | /v1/runs | upload a file (+ optional recipe) -> run summary; artifacts stored server-side |
| GET  | /v1/runs, /v1/runs/{id} | list runs / one run summary |
| GET  | /v1/runs/{id}/files/{name} | download an artifact (data.csv, data.parquet, report.html, datacard.md, ...) |
| POST | /v1/runs/{id}/transform | apply that run's fitted pipeline to a new upload |

If ``FF_API_KEY`` is set every ``/v1`` call needs the header ``X-API-Key``. Uploaded pipelines are never unpickled: the
transform endpoint only uses pipelines the server created itself.
"""

import logging
import secrets
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from . import __version__
from .auto import auto_recipe
from .config import get_settings
from .io import SUPPORTED_READ, TableReadError, detect_format, read_table, write_table
from .log import setup_logging
from .pipeline import Pipeline, PipelineError
from .profile import profile as run_profile
from .recipe import Recipe
from .service import RunStore, run_job
from .steps import catalog

logger = logging.getLogger(__name__)
CHUNK = 1 << 20
MEDIA = {
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".html": "text/html",
    ".md": "text/markdown",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".zip": "application/zip",
    ".joblib": "application/octet-stream",
}


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


async def _read_upload(file: UploadFile) -> tuple:
    """Return (DataFrame, bytes) with size limit and readable errors."""
    name = file.filename or ""
    try:
        detect_format(name)
    except TableReadError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    limit = get_settings().max_upload_bytes
    buf = bytearray()
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail=f"file larger than {get_settings().max_upload_mb} MB")
    try:
        df = read_table(bytes(buf), name=name)
    except TableReadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return df, bytes(buf)


def _check_policy(policy: str) -> None:
    if policy not in ("auto", "drop", "warn"):
        raise HTTPException(status_code=422, detail="leak_policy must be auto, drop or warn")


def _opt(v: Optional[str]) -> Optional[str]:
    return v if v not in (None, "", "null", "none") else None


def create_app() -> FastAPI:
    setup_logging()
    api = FastAPI(
        title="FeatureForge API",
        version=__version__,
        description="Data cleaning, profiling and feature engineering for tabular files.",
    )
    v1 = Depends(require_key)

    @api.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "version": __version__}

    @api.get("/v1/info", dependencies=[v1])
    def info() -> Dict[str, Any]:
        s = get_settings()
        return {
            "version": __version__,
            "max_upload_mb": s.max_upload_mb,
            "profile_sample_rows": s.profile_sample_rows,
            "formats": sorted(SUPPORTED_READ),
        }

    @api.get("/v1/steps", dependencies=[v1])
    def steps() -> list:
        return catalog()

    @api.post("/v1/profile", dependencies=[v1])
    async def profile_endpoint(file: UploadFile = File(...), target: Optional[str] = Form(None)) -> Dict[str, Any]:
        df, _ = await _read_upload(file)
        try:
            return run_profile(df, target=_opt(target))
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.post("/v1/recipes/auto", dependencies=[v1])
    async def recipe_auto(
        file: UploadFile = File(...),
        target: Optional[str] = Form(None),
        level: str = Form("standard"),
        leak_policy: str = Form("auto", description="auto | drop | warn"),
    ) -> Dict[str, Any]:
        if level not in ("clean", "standard", "full"):
            raise HTTPException(status_code=422, detail="level must be clean, standard or full")
        _check_policy(leak_policy)
        df, _ = await _read_upload(file)
        try:
            r = auto_recipe(df, target=_opt(target), level=level, leak_policy=leak_policy)  # type: ignore[arg-type]
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"recipe": r.to_dict(), "yaml": r.to_yaml()}

    @api.post("/v1/recipes/validate", dependencies=[v1])
    def recipe_validate(recipe: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = Recipe.model_validate(recipe)
        except Exception as exc:
            return {"valid": False, "errors": [str(exc)]}
        errors = r.validate_steps()
        return {"valid": not errors, "errors": errors}

    @api.post("/v1/runs", dependencies=[v1])
    async def create_run(
        file: UploadFile = File(...),
        recipe: Optional[str] = Form(None, description="recipe as YAML or JSON text; omit for auto mode"),
        target: Optional[str] = Form(None),
        level: str = Form("standard"),
        test_size: Optional[float] = Form(None),
        split_strategy: str = Form("auto"),
        group_column: Optional[str] = Form(None),
        time_column: Optional[str] = Form(None),
        leak_policy: str = Form("auto", description="auto mode only: auto | drop | warn"),
    ) -> Dict[str, Any]:
        _check_policy(leak_policy)
        df, raw = await _read_upload(file)
        rec = None
        if _opt(recipe):
            try:
                rec = Recipe.from_text(recipe)  # type: ignore[arg-type]
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"invalid recipe: {exc}") from exc
            errors = rec.validate_steps()
            if errors:
                raise HTTPException(status_code=422, detail=errors)
        store = RunStore()
        run_id, d = store.new()
        src = d / f"input{Path(file.filename or 'x.csv').suffix.lower()}"
        src.write_bytes(raw)
        try:
            res = run_job(
                df,
                out_dir=d,
                recipe=rec,
                target=_opt(target),
                level=level,
                test_size=test_size or None,
                split_strategy=split_strategy,
                group_column=_opt(group_column),
                time_column=_opt(time_column),
                source_name=Path(file.filename or "upload").name,
                source_path=src,
                run_id=run_id,
                leak_policy=leak_policy,
            )
        except (PipelineError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return res.summary

    @api.get("/v1/runs", dependencies=[v1])
    def list_runs() -> list:
        return [
            {k: r.get(k) for k in ("run_id", "source", "created", "input_shape", "output_shape", "task")}
            for r in RunStore().list()
        ]

    @api.get("/v1/runs/{run_id}", dependencies=[v1])
    def get_run(run_id: str) -> Dict[str, Any]:
        import json

        try:
            return json.loads((RunStore().path(run_id) / "summary.json").read_text(encoding="utf-8"))
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @api.get("/v1/runs/{run_id}/files/{name}", dependencies=[v1])
    def get_file(run_id: str, name: str) -> FileResponse:
        import json

        try:
            d = RunStore().path(run_id)
            files = json.loads((d / "summary.json").read_text(encoding="utf-8"))["files"]
            if name not in files:
                raise KeyError(name)
            path = RunStore().file(run_id, files[name])
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=f"file {name!r} not found") from exc
        return FileResponse(path, media_type=MEDIA.get(path.suffix, "application/octet-stream"), filename=name)

    @api.post("/v1/runs/{run_id}/transform", dependencies=[v1])
    async def transform_new(run_id: str, file: UploadFile = File(...), format: str = Form("csv")) -> FileResponse:
        if format not in ("csv", "parquet", "json"):
            raise HTTPException(status_code=422, detail="format must be csv, parquet or json")
        try:
            d = RunStore().path(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        df, _ = await _read_upload(file)
        try:
            pipe = Pipeline.load(d / "pipeline")  # created by this server, never uploaded
            out: pd.DataFrame = pipe.transform(df)
        except PipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        tmp = Path(tempfile.mkdtemp(prefix="ff_")) / f"transformed.{format}"
        write_table(out, tmp)
        headers = {"X-FeatureForge-Warnings": "; ".join(pipe.warnings_)[:1000]} if pipe.warnings_ else None
        return FileResponse(tmp, media_type=MEDIA.get(f".{format}"), filename=tmp.name, headers=headers)

    landing = "/docs"
    try:
        import gradio as gr

        from .dashboard import DASHBOARD_CSS, build_dashboard

        gr.mount_gradio_app(
            api,
            build_dashboard(),
            path="/ui",
            css=DASHBOARD_CSS,
            theme=gr.themes.Soft(),
            max_file_size=f"{get_settings().max_upload_mb}mb",
        )
        landing = "/ui"
    except Exception:  # the API must keep working without the UI
        logger.exception("dashboard not mounted")

    @api.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url=landing)

    return api


app = create_app()
