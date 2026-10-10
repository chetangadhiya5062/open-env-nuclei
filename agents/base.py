"""Common agent interface."""

from abc import ABC, abstractmethod

from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation


class Agent(ABC):
    """An agent maps an observation to the next action. Agents are stateful within an episode."""

    name: str = "agent"

    #: how many times the agent had to fall back from its own decision (reported by the benchmark)
    fallbacks: int = 0
    #: chat-API calls that failed even after retries (LLM agent only)
    api_errors: int = 0

    def reset(self, seed: int = 0) -> None:
        """Called at the start of every episode."""
        self.fallbacks = 0
        self.api_errors = 0

    @abstractmethod
    def act(self, obs: DataCleaningObservation) -> DataCleaningAction: ...
