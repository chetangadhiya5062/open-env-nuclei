"""Target schema of the synthetic *customers* table.

The schema is part of the task description and is shown to the agent through the observation
(``expected_type``, ``allowed_values``, ``valid_range``), like column types and CHECK constraints
in a real database. It is what lets the environment say objectively whether a cell is "clean".
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Kind = Literal["id", "text", "cat", "int", "float", "date"]

CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
PLANS = ["basic", "pro", "enterprise"]
ISO_DATE_FORMAT = "%Y-%m-%d"
# String tokens that mean "no value" but are not real nulls.
PLACEHOLDERS = frozenset({"", "n/a", "na", "null", "none", "nan", "-", "?"})


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: Kind
    allowed: Optional[Tuple[str, ...]] = None
    valid_range: Optional[Tuple[float, float]] = None

    @property
    def is_numeric(self) -> bool:
        return self.kind in ("int", "float")


SCHEMA = (
    ColumnSpec("customer_id", "id"),
    ColumnSpec("name", "text"),
    ColumnSpec("age", "int", valid_range=(0, 120)),
    ColumnSpec("city", "cat", allowed=tuple(CITIES)),
    ColumnSpec("signup_date", "date"),
    ColumnSpec("plan", "cat", allowed=tuple(PLANS)),
    ColumnSpec("monthly_spend", "float", valid_range=(0, 1000)),
)
SCHEMA_BY_NAME = {c.name: c for c in SCHEMA}
COLUMNS = [c.name for c in SCHEMA]
ID_COLUMN = "customer_id"
