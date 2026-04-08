# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data Cleaning Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DataCleaningAction, DataCleaningObservation


class DataCleaningEnv(
    EnvClient[DataCleaningAction, DataCleaningObservation, State]
):
    """
    Client for the Data Cleaning Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DataCleaningEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(DataCleaningAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DataCleaningEnv.from_docker_image("data_cleaning_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DataCleaningAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DataCleaningAction) -> Dict:
        """
        Convert DataCleaningAction to JSON payload for step message.

        Args:
            action: DataCleaningAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
                "action_type": action.action_type,
                "column_name": action.column_name,
                "value": action.value,
            }

    def _parse_result(self, payload: Dict) -> StepResult[DataCleaningObservation]:
        """
        Parse server response into StepResult[DataCleaningObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DataCleaningObservation
        """
        obs_data = payload.get("observation", {})
        observation = DataCleaningObservation(
            missing_values_count_per_column=obs_data.get("missing_values_count_per_column", {}),
            duplicate_row_count=obs_data.get("duplicate_row_count", 0),
            total_row_count=obs_data.get("total_row_count", 0),
            column_names=obs_data.get("column_names", []),
            
            data_sample = obs_data.get("data_sample", []),
            
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
