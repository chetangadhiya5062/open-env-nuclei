"""Client for the Data Cleaning environment.

Example (the server must be running, see README):

    from data_cleaning_env import DataCleaningAction, DataCleaningEnv

    with DataCleaningEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="medium-clean", seed=3)
        result = env.step(DataCleaningAction(action_type="drop_duplicates"))
        print(result.reward, result.observation.quality_score)
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import DataCleaningAction, DataCleaningObservation, DataCleaningState


class DataCleaningEnv(EnvClient[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """WebSocket client: each instance owns one stateful episode on the server."""

    def _step_payload(self, action: DataCleaningAction) -> Dict:
        return action.to_payload()

    def _parse_result(self, payload: Dict) -> StepResult[DataCleaningObservation]:
        obs = DataCleaningObservation(
            **payload.get("observation", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> DataCleaningState:
        return DataCleaningState(**payload)
