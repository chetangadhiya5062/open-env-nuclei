"""Pipeline steps. Importing this package registers every step in ``REGISTRY``."""

from . import cleaning, encoding, features, selection  # noqa: F401  (registration side effects)
from .base import REGISTRY, FitContext, Step, StepError


def catalog() -> list:
    """Name, kind, description and JSON schema of the parameters of every step (for UIs and docs)."""
    return [
        {
            "step": name,
            "kind": cls.kind,
            "description": cls.description,
            "train_only": cls.train_only,
            "needs_target": cls.needs_target,
            "params": cls.Params.model_json_schema(),
        }
        for name, cls in REGISTRY.items()
    ]


__all__ = ["REGISTRY", "FitContext", "Step", "StepError", "catalog"]
