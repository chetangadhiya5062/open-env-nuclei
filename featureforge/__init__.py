"""FeatureForge: reproducible data cleaning and feature engineering for tabular data.

Typical use::

    from featureforge import Pipeline, auto_recipe, read_table

    df = read_table("customers.csv")
    recipe = auto_recipe(df, target="churn")
    pipe = Pipeline(recipe)
    out = pipe.fit_transform(df)          # learn on training data
    pipe.save("artifacts/churn")          # recipe.yaml + pipeline.joblib + metadata.json
    new = Pipeline.load("artifacts/churn").transform(read_table("new_customers.csv"))
"""

from .auto import auto_recipe
from .io import read_table, write_table
from .pipeline import Pipeline, PipelineError
from .profile import drift_report, profile
from .recipe import Recipe, StepSpec
from .split import leakage_safe_split
from .types import ColumnType, infer_types

__version__ = "0.4.1"

__all__ = [
    "ColumnType",
    "Pipeline",
    "PipelineError",
    "Recipe",
    "StepSpec",
    "auto_recipe",
    "drift_report",
    "infer_types",
    "leakage_safe_split",
    "profile",
    "read_table",
    "write_table",
    "__version__",
]
