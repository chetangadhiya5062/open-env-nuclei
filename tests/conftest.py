import pytest

from data_cleaning_env.models import DataCleaningAction
from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment


@pytest.fixture
def env():
    return DataCleaningEnvironment()


def act(**kwargs) -> DataCleaningAction:
    return DataCleaningAction(**kwargs)
