"""Logging setup shared by the CLI, API and dashboard."""

import logging
from typing import Optional

from .config import get_settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: Optional[str] = None) -> None:
    """Configure the root logger once (idempotent)."""
    lvl = (level or get_settings().log_level).upper()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=lvl, format=_FORMAT)
    root.setLevel(lvl)
    for noisy in ("httpx", "matplotlib", "PIL", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
