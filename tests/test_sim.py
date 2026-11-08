import numpy as np
import pytest

from robot_policy_lab.sim import PushEnv, expert_action


def rollout(seed):
    env = PushEnv()
    observation = env.reset(seed)
    states = [observation]
    done = False
    while not done:
        observation, success, done = env.step(expert_action(observation))
        states.append(observation)
    return np.asarray(states), success


def test_reset_determinism_and_seed_variation():
    env = PushEnv()
    first = env.reset(123)
    env.step(np.ones(2))
    np.testing.assert_array_equal(env.reset(123), first)
    assert not np.array_equal(first, env.reset(124))
    assert first.shape == (12,)


def test_rollout_determinism():
    first, _ = rollout(42)
    second, _ = rollout(42)
    np.testing.assert_array_equal(first, second)


def test_lateral_stress_changes_only_initial_arm_position():
    baseline = PushEnv().reset(17)
    stress = PushEnv(lateral_offset=0.12).reset(17)
    assert stress[1] - baseline[1] == pytest.approx(0.12)
    np.testing.assert_array_equal(stress[2:6], baseline[2:6])
    np.testing.assert_array_equal(stress, PushEnv(lateral_offset=0.12).reset(17))
    with pytest.raises(ValueError, match="lateral_offset"):
        PushEnv(lateral_offset=float("nan"))
    with pytest.raises(ValueError, match="lateral_offset"):
        PushEnv(lateral_offset=0.16)


@pytest.mark.parametrize("seed", [*range(12), 16, 21, 42, 124])
def test_expert_succeeds(seed):
    states, success = rollout(seed)
    assert success, f"expert failed seed {seed}, final delta {states[-1, 8:10]}"
    assert np.linalg.norm(states[-1, 8:10]) < 0.025
    assert len(states) <= 121


@pytest.mark.parametrize("action", [[1, 2], [np.nan, 0], [1], [[0, 0]]])
def test_bad_action(action):
    env = PushEnv()
    env.reset(0)
    with pytest.raises(ValueError, match="action"):
        env.step(action)


def test_episode_lifecycle():
    with pytest.raises(ValueError, match="horizon"):
        PushEnv(0)
    env = PushEnv(horizon=1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(2))
    env.reset()
    _, success, done = env.step(np.zeros(2))
    assert done and not success
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(2))
