"""Demonstrates VecStorm through the optional Gymnasium-compatible wrappers in
`vec_storm.gymnasium_env`.

Since `StormVecEnv` already runs many environment copies in lockstep, `StormGymVecEnv`
implements the `gymnasium.vector.VectorEnv` API and is the natural fit; `StormGymEnv`
is a thin `num_envs=1` convenience wrapper for tooling that expects a plain
`gymnasium.Env`. Both are demonstrated below.

Requires the optional `gymnasium` dependency: `pip install vec_storm[gym]`.

Usage:
    python demo_gymnasium.py
"""
import os

import numpy as np
import paynt.parser.sketch

from vec_storm import StormVecEnv
from vec_storm.gymnasium_env import StormGymVecEnv, StormGymEnv


MODEL_PATH = os.path.join(os.path.dirname(__file__), "vec_storm", "tests", "models", "det_avoid")


def load_pomdp(model_path):
    sketch_path = os.path.join(model_path, "sketch.templ")
    properties_path = os.path.join(model_path, "sketch.props")
    quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return quotient.pomdp


def scalarize_reward(rewards, reward_types):
    return rewards[reward_types[-1]]


def sample_masked_actions(action_mask, rng):
    """Samples one uniformly random allowed action per environment from a batched mask."""
    num_envs = action_mask.shape[0]
    actions = np.zeros(num_envs, dtype=np.int64)
    for i in range(num_envs):
        legal = np.flatnonzero(action_mask[i])
        actions[i] = rng.choice(legal)
    return actions


def run_vector_env(pomdp, metalabels):
    print("=== gymnasium.vector.VectorEnv (StormGymVecEnv) ===")
    num_envs = 4
    storm_env = StormVecEnv(pomdp, scalarize_reward, seed=42, num_envs=num_envs, max_steps=50, metalabels=metalabels)
    env = StormGymVecEnv(storm_env)

    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")

    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)

    total_reward = np.zeros(num_envs, dtype=np.float32)
    for step in range(20):
        actions = sample_masked_actions(obs["action_mask"], rng)
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += rewards
        if (terminated | truncated).any():
            print(f"step={step:2d}  terminated={terminated}  truncated={truncated}")

    print(f"Cumulative reward after 20 steps per env: {total_reward}")
    env.close()


def run_single_env(pomdp, metalabels):
    print()
    print("=== plain gymnasium.Env (StormGymEnv, num_envs=1) ===")
    storm_env = StormVecEnv(pomdp, scalarize_reward, seed=42, num_envs=1, max_steps=50, metalabels=metalabels)
    env = StormGymEnv(storm_env)

    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")

    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)

    episode_return = 0.0
    for step in range(20):
        legal = np.flatnonzero(obs["action_mask"])
        action = rng.choice(legal)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        if terminated or truncated:
            print(f"step={step:2d}  episode ended (terminated={terminated}, truncated={truncated}), "
                  f"return so far={episode_return:.2f}")
    env.close()


def main():
    pomdp = load_pomdp(MODEL_PATH)
    metalabels = {"avoid": ["traps"], "reach": ["goal"]}
    run_vector_env(pomdp, metalabels)
    run_single_env(pomdp, metalabels)


if __name__ == "__main__":
    main()
