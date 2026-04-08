# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data Cleaning Env Environment.
"""

from typing import Dict, List, Optional, Literal, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DataCleaningAction(Action):
    """
    Action for Data Cleaning Environment.
    """

    action_type: Literal[
        "fill_missing",
        "drop_rows_with_missing",
        "remove_duplicates",
        "finish_cleaning"
    ]

    column_name: Optional[str] = None
    value: Optional[str] = None


class DataCleaningObservation(Observation):

    missing_values_count_per_column: Dict[str, int]
    duplicate_row_count: int
    total_row_count: int
    column_names: List[str]

    # ✅ NEW FIELD (Phase 12)
    data_sample: List[Dict[str, Any]]

    reward: float = 0.0
    done: bool = False