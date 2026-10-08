"""FastAPI application for the Data Cleaning environment.

Endpoints (provided by OpenEnv's ``create_app``):
    GET  /health   liveness probe
    GET  /schema   action / observation JSON schemas
    POST /reset    stateless one-shot reset (each HTTP call builds a fresh env)
    POST /step     stateless one-shot step
    WS   /ws       persistent session: the one to use for real episodes

Run locally:
    python -m data_cleaning_env.server.app            # port from $PORT, default 7860
    uvicorn data_cleaning_env.server.app:app --port 7860
"""

import logging
import os

from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_app

from ..models import DataCleaningAction, DataCleaningObservation
from .data_cleaning_env_environment import DataCleaningEnvironment

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7860

app = create_app(
    DataCleaningEnvironment,
    DataCleaningAction,
    DataCleaningObservation,
    env_name="data_cleaning_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
)


def _mount_demo() -> str:
    """Mount the Gradio demo at /demo when gradio is installed; return the landing path."""
    try:
        import gradio as gr

        from ..ui import build_demo

        gr.mount_gradio_app(app, build_demo(), path="/demo")
        return "/demo"
    except Exception:  # the API must keep working without the optional UI
        logger.exception("Gradio demo not mounted")
        return "/docs"


LANDING = _mount_demo()


def _root() -> RedirectResponse:
    return RedirectResponse(url=LANDING)


app.router.add_api_route("/", _root, methods=["GET"], include_in_schema=False)


def main() -> None:
    """Entry point for ``python -m data_cleaning_env.server.app``."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Data Cleaning OpenEnv server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", DEFAULT_PORT)))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Starting server on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
