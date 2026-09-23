
from vec_storm.storm_vec_env import StormVecEnv
from vec_storm.tests.test_tools import generate_reward_selection_function, load_pomdp
from vec_storm.trajectory_checker import TrajectoryChecker

import numpy as np

def test_check_trajectory_true():
    """
        Test that the trajectory checker returns True for a valid trajectory.
    """
    env = StormVecEnv(load_pomdp("vec_storm/tests/models/network-3-8-20"), generate_reward_selection_function, num_envs=1);
    trajectory_checker = TrajectoryChecker(env)
    _, allowed_actions, _ = env.reset()
    observation_integers = [env.simulator_integer_observations[0, 0].item()]
    nr_steps = 20
    for i in range(nr_steps):
        action = np.random.choice(np.where(allowed_actions[0] == 1)[0])
        _, _, _, _, allowed_actions, _ = env.step(np.array([action]))
        integer_observations = env.simulator_integer_observations
        observation_integers.append(integer_observations[0, 0].item())

    assert trajectory_checker.check_trajectory(observation_integers)

def test_check_trajectory_false():
    env = StormVecEnv(load_pomdp("vec_storm/tests/models/network-3-8-20"), generate_reward_selection_function, num_envs=1);
    trajectory_checker = TrajectoryChecker(env)
    _, allowed_actions, _ = env.reset()
    observation_integers = [env.simulator_integer_observations[0, 0].item()]
    nr_steps = 20
    for i in range(nr_steps):
        action = np.random.choice(np.where(allowed_actions[0] == 1)[0])
        _, _, _, _, allowed_actions, _ = env.step(np.array([action]))
        if i == 10:
            _, allowed_actions, _ = env.reset()
        integer_observations = env.simulator_integer_observations
        observation_integers.append(integer_observations[0, 0].item())

    assert not trajectory_checker.check_trajectory(observation_integers)


