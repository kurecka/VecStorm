"""Demonstrates the "native" way of using `StormVecEnv`, i.e. talking to it directly
through its `reset()` / `step()` tuple API. This mirrors how VecStorm is used inside
`rl_synthesis`'s `EnvironmentWrapperVec` (see
`compact_rl/rl/environment/environment_wrapper_vec.py`), just stripped down to plain
numpy so it runs with only the `vec_storm` package installed - no TF-Agents/TensorFlow
required.

For a Gymnasium-style interface instead, see `demo_gymnasium.py`.

Usage:
    python demo.py
"""
import os

import numpy as np
import paynt.parser.sketch

from vec_storm import StormVecEnv


MODEL_PATH = os.path.join(os.path.dirname(__file__), "vec_storm", "tests", "models", "det_avoid")


def load_pomdp(model_path):
    """Loads a PRISM sketch (`sketch.templ` + `sketch.props`) into a stormpy POMDP.

    This is the same loading step `EnvironmentWrapperVec` performs before handing the
    POMDP to `StormVecEnv` - it is entirely independent of VecStorm itself, which only
    needs a stormpy POMDP object, however it is constructed.
    """
    sketch_path = os.path.join(model_path, "sketch.templ")
    properties_path = os.path.join(model_path, "sketch.props")
    quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return quotient.pomdp


def scalarize_reward(rewards, reward_types):
    """Picks out the last reward signal defined in the model, matching
    `generate_reward_selection_function` from `environment_wrapper_vec.py`.
    """
    return rewards[reward_types[-1]]


def sample_allowed_actions(allowed_actions, rng):
    """Samples one uniformly random *allowed* action per environment.

    Equivalent in spirit to `EnvironmentWrapperVec.change_illegal_actions_to_random_allowed`,
    which redirects any illegal action a policy proposes towards a random legal one.
    """
    num_envs, num_actions = allowed_actions.shape
    actions = np.zeros(num_envs, dtype=np.int32)
    for i in range(num_envs):
        legal = np.flatnonzero(allowed_actions[i])
        actions[i] = rng.choice(legal)
    return actions


def main():
    num_envs = 8
    max_steps = 50
    num_steps = 200

    pomdp = load_pomdp(MODEL_PATH)

    # Metalabels let VecStorm precompute boolean conjunctions of atomic labels, e.g. to
    # flag goal/failure states. Since a metalabel is now also folded into `done`
    # (see simulator.py), reaching "reach" or "avoid" ends the episode even on models
    # where those states aren't themselves POMDP sinks.
    metalabels = {"avoid": ["traps"], "reach": ["goal"]}

    env = StormVecEnv(pomdp, scalarize_reward, seed=42, num_envs=num_envs, max_steps=max_steps, metalabels=metalabels)

    print(f"Action labels: {env.get_action_labels()}")
    print(f"Observation labels: {env.get_observation_labels()}")

    rng = np.random.default_rng(0)
    observations, allowed_actions, metalabel_flags = env.reset()

    episode_return = np.zeros(num_envs, dtype=np.float32)
    completed_returns = []
    reached_goal = 0
    hit_trap = 0

    for step in range(num_steps):
        actions = sample_allowed_actions(np.asarray(allowed_actions), rng)
        observations, rewards, done, truncated, allowed_actions, metalabel_flags = env.step(actions)

        episode_return += np.asarray(rewards)
        done = np.asarray(done)
        metalabel_flags = np.asarray(metalabel_flags)

        for i in np.flatnonzero(done):
            completed_returns.append(episode_return[i])
            # metalabel_flags columns follow the order of `metalabels` above: [avoid, reach]
            hit_trap += int(metalabel_flags[i, 0])
            reached_goal += int(metalabel_flags[i, 1])
            episode_return[i] = 0.0

        if step % 50 == 0:
            print(f"step={step:4d}  mean live return={episode_return.mean():.2f}  "
                  f"completed episodes={len(completed_returns)}")

    print()
    print(f"Completed episodes: {len(completed_returns)}")
    if completed_returns:
        print(f"Mean episode return: {np.mean(completed_returns):.2f}")
    print(f"Reached goal: {reached_goal}, hit trap: {hit_trap}")


if __name__ == "__main__":
    main()
