"""Declarative recipes: the reproducible description of a pipeline (saved as YAML or JSON).

```yaml
version: 1
name: churn-prep
target: churned
task: auto              # auto | classification | regression | none
id_columns: [customer_id]
type_overrides: {zip_code: categorical}
random_state: 42
steps:
  - step: normalize_strings
  - step: fix_types
  - step: impute
    params: {numeric_strategy: median, add_indicator: true}
  - step: one_hot
    params: {max_categories: 15}
```

A recipe holds no fitted state. Fitting it on data produces a :class:`featureforge.pipeline.Pipeline` artifact.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

RECIPE_VERSION = 1


class StepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str = Field(description="registered step name, e.g. 'impute'")
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = RECIPE_VERSION
    name: str = "recipe"
    description: str = ""
    target: Optional[str] = None
    task: Literal["auto", "classification", "regression", "none"] = "auto"
    id_columns: List[str] = Field(default_factory=list)
    type_overrides: Dict[str, str] = Field(default_factory=dict)
    random_state: int = 42
    steps: List[StepSpec] = Field(default_factory=list)

    # ----------------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_text(cls, text: str) -> "Recipe":
        """Parse YAML or JSON (JSON is valid YAML)."""
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("a recipe must be a mapping with a 'steps' list")
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Recipe":
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() if path.suffix == ".json" else self.to_yaml(), encoding="utf-8")
        return path

    def validate_steps(self) -> List[str]:
        """Errors for unknown steps or invalid parameters (empty list = valid)."""
        from .steps import REGISTRY

        errors = []
        for i, spec in enumerate(self.steps):
            cls = REGISTRY.get(spec.step)
            if cls is None:
                errors.append(f"step {i + 1}: unknown step {spec.step!r}; available: {', '.join(sorted(REGISTRY))}")
                continue
            try:
                cls.Params(**spec.params)
            except Exception as exc:
                errors.append(f"step {i + 1} ({spec.step}): {exc}")
        return errors
