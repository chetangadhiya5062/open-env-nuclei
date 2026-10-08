"""Common agent interface."""

from abc import ABC, abstractmethod

from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation


class Agent(ABC):
    """An agent maps an observation to the next action. Agents are stateful within an episode."""

    name: str = "agent"

    #: how many times the agent had to fall back from its own decision (reported by the benchmark)
    fallbacks: int = 0

    def reset(self, seed: int = 0) -> None:
        """Called at the start of every episode."""
        self.fallbacks = 0

    @abstractmethod
    def act(self, obs: DataCleaningObservation) -> DataCleaningAction:
        ...
