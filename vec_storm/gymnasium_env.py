"""Gymnasium-compatible wrappers around :class:`~vec_storm.storm_vec_env.StormVecEnv`.

VecStorm already runs many environment instances in lockstep (via JAX), so the natural
Gymnasium counterpart is the vectorized ``gymnasium.vector.VectorEnv`` API rather than a
single ``gymnasium.Env``. :class:`StormGymVecEnv` adapts the
``(observations, rewards, done, truncated, allowed_actions, metalabels)`` tuples used by
:class:`StormVecEnv` to that interface. :class:`StormGymEnv` is a thin convenience wrapper
around it for callers that only want a single, non-vectorized environment (``num_envs=1``)
and expect a plain ``gymnasium.Env``.

This module requires the optional ``gymnasium`` dependency (``pip install vec_storm[gym]``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.vector.vector_env import AutoresetMode

from .storm_vec_env import StormVecEnv


class StormGymVecEnv(VectorEnv):
    """Adapts :class:`StormVecEnv` to the Gymnasium vector-environment API.

    Observations are exposed as a ``Dict`` space with an ``"observation"`` entry (the
    valuation-based float observation returned by VecStorm) and an ``"action_mask"``
    entry (the boolean mask of allowed actions), mirroring the observation/mask pattern
    used by ``rl_synthesis``'s ``EnvironmentWrapperVec``.

    VecStorm resets a sub-environment on the step *after* it terminates or is truncated
    (the observation returned on the terminal step is still the real terminal
    observation, not the reset one), which matches Gymnasium's default ``"next-step"``
    autoreset convention, so no extra bookkeeping is required here.
    """

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, storm_vec_env: StormVecEnv, render_mode: Optional[str] = None):
        self.storm_vec_env = storm_vec_env
        self.render_mode = render_mode
        self.num_envs = int(storm_vec_env.simulator_states.vertices.shape[0])

        nr_actions = len(storm_vec_env.get_action_labels())
        obs_dim = int(storm_vec_env.simulator.observations.shape[-1])

        self.single_observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(nr_actions),
        })
        self.single_action_space = spaces.Discrete(nr_actions)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    @staticmethod
    def _pack_obs(observations, allowed_actions) -> Dict[str, np.ndarray]:
        return {
            "observation": np.asarray(observations, dtype=np.float32),
            "action_mask": np.asarray(allowed_actions, dtype=np.int8),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.storm_vec_env.set_seed(seed)
        observations, allowed_actions, metalabels = self.storm_vec_env.reset()
        obs = self._pack_obs(observations, allowed_actions)
        info = {"metalabels": np.asarray(metalabels)}
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        observations, rewards, done, truncated, allowed_actions, metalabels = self.storm_vec_env.step(
            np.asarray(actions)
        )
        done = np.asarray(done)
        truncations = np.asarray(truncated)
        # StormVecEnv's `done` already folds truncation in; Gymnasium wants the two
        # kept apart, with `terminated` reserved for genuine (non-time-limit) endings.
        terminations = done & ~truncations
        obs = self._pack_obs(observations, allowed_actions)
        infos = {"metalabels": np.asarray(metalabels)}
        return obs, np.asarray(rewards, dtype=np.float32), terminations, truncations, infos


class StormGymEnv(gym.Env):
    """A single, non-vectorized Gymnasium environment backed by :class:`StormVecEnv`.

    Convenience wrapper for tooling that expects a plain ``gymnasium.Env`` (e.g.
    ``gymnasium.utils.env_checker.check_env`` or single-environment training loops).
    It is implemented on top of :class:`StormGymVecEnv` with ``num_envs=1`` and simply
    strips the batch dimension from observations, rewards, infos, etc.
    """

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, storm_vec_env: StormVecEnv, render_mode: Optional[str] = None):
        if int(storm_vec_env.simulator_states.vertices.shape[0]) != 1:
            raise ValueError("StormGymEnv requires a StormVecEnv created with num_envs=1.")
        self._vec_env = StormGymVecEnv(storm_vec_env, render_mode=render_mode)
        self.observation_space = self._vec_env.single_observation_space
        self.action_space = self._vec_env.single_action_space
        self.render_mode = render_mode

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self._vec_env.reset(seed=seed, options=options)
        return {k: v[0] for k, v in obs.items()}, {k: v[0] for k, v in info.items()}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._vec_env.step(np.asarray([action]))
        obs = {k: v[0] for k, v in obs.items()}
        info = {k: v[0] for k, v in info.items()}
        return obs, float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

    def close(self):
        pass
