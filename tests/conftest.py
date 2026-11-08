import pytest

from robot_policy_lab.data import collect_episode, write_dataset


@pytest.fixture(scope="session")
def episodes():
    return [collect_episode(seed) for seed in range(6)]


@pytest.fixture
def dataset(tmp_path, episodes):
    return write_dataset(episodes, tmp_path / "data")
