# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data Cleaning Env Environment.
"""

from typing import Dict, List, Optional, Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DataCleaningAction(Action):
    """
    Action for Data Cleaning Environment.
    """

    action_type: Literal[
        "fill_missing",
        "drop_rows_with_missing",
        "remove_duplicates"
    ]

    column_name: Optional[str] = None
    value: Optional[str] = None


class DataCleaningObservation(Observation):
    """
    Observation for Data Cleaning Environment.
    """

    missing_values_count_per_column: dict
    duplicate_row_count: int
    total_row_count: int
    column_names: list

    reward: float = 0.0   # ✅ REQUIRED
    done: bool = False