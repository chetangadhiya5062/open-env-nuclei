"""Runtime settings, read from environment variables (prefix ``FF_``) with safe defaults.

| variable | default | meaning |
|---|---|---|
| ``FF_WORK_DIR`` | ``./ff_workspace`` | where the API / dashboard store runs and artifacts |
| ``FF_MAX_UPLOAD_MB`` | ``200`` | largest accepted upload |
| ``FF_PROFILE_SAMPLE_ROWS`` | ``50000`` | profiling works on a random sample of this many rows |
| ``FF_MAX_FIT_ROWS`` | ``20000`` | cap for expensive fits (KNN / iterative imputation, isolation forest, MI, generation) |
| ``FF_PREVIEW_ROWS`` | ``20`` | rows shown in previews |
| ``FF_RANDOM_STATE`` | ``42`` | default seed |
| ``FF_LOG_LEVEL`` | ``INFO`` | logging level |
| ``FF_API_KEY`` | unset | if set, the REST API requires header ``X-API-Key`` |
| ``FF_RUN_TTL_HOURS`` | ``72`` | runs older than this are deleted by ``cleanup_runs`` |
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"environment variable {name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    work_dir: Path = field(default_factory=lambda: Path(os.getenv("FF_WORK_DIR", "ff_workspace")))
    max_upload_mb: int = field(default_factory=lambda: _int("FF_MAX_UPLOAD_MB", 200))
    profile_sample_rows: int = field(default_factory=lambda: _int("FF_PROFILE_SAMPLE_ROWS", 50_000))
    max_fit_rows: int = field(default_factory=lambda: _int("FF_MAX_FIT_ROWS", 20_000))
    preview_rows: int = field(default_factory=lambda: _int("FF_PREVIEW_ROWS", 20))
    random_state: int = field(default_factory=lambda: _int("FF_RANDOM_STATE", 42))
    log_level: str = field(default_factory=lambda: os.getenv("FF_LOG_LEVEL", "INFO"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("FF_API_KEY") or None)
    run_ttl_hours: int = field(default_factory=lambda: _int("FF_RUN_TTL_HOURS", 72))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    """Fresh settings from the current environment (cheap; call where needed so tests can monkeypatch env vars)."""
    return Settings()
