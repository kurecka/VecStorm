from typing import Set
import json
import pickle

import numpy as np

import jax
from jax import numpy as jnp

from stormpy import simulator
from stormpy.storage.storage import SparsePomdp

from .sparse_array import SparseArray
from .simulator import Simulator, States, StepInfo, ResetInfo

import logging
logger = logging.getLogger(__name__)


def cast2jax(data):
    if isinstance(data, SparseArray):
        return SparseArray(
            nr_actions = data.nr_actions,
            nr_states = data.nr_states,
            nr_rows = data.nr_rows,
            row_starts = jnp.array(data.row_starts),
            row_ends = jnp.array(data.row_ends),
            indices = jnp.array(data.indices),
            data = jnp.array(data.data)
        )
    if isinstance(data, np.ndarray):
        return jnp.array(data)


class StormVecEnvBuilder:
    NO_LABEL = "__no_label__"

    @classmethod
    def get_action_labels(cls, pomdp, ignore_label) -> Set[str]:
        action_labels = set()
        for state in range(pomdp.nr_states):
            n_act = pomdp.get_nr_available_actions(state)
            for action in range(n_act):
                for label in pomdp.choice_labeling.get_labels_of_choice(pomdp.get_choice_index(state, action)):
                    action_labels.add(label)
        if ignore_label in action_labels:
            action_labels.remove(ignore_label)
        return list(sorted(action_labels))

    @classmethod
    def build_from_pomdp(cls, pomdp: SparsePomdp, get_scalarized_reward, num_envs=1, seed=42, metalabels=None, max_steps=100, random_init=False):
        """
            pomdp: The POMDP object that should be compiled into a jax-based environment.
            get_scalarized_reward: A function that accepts a dictionary indexed by reward signal names and returns a number.
        """

        sim = simulator.create_simulator(pomdp)
        action_labels = cls.get_action_labels(pomdp, cls.NO_LABEL)
        action_labels2indices = {label: i for i, label in enumerate(action_labels)}
        initial_state = pomdp.initial_states[0]

        rewards_types = sim.get_reward_names()
        nr_states = pomdp.nr_states
        nr_actions = len(action_labels)

        # Row map: Assigns each (state, action) pair a row in the spare transition/reward matrix
        # or -1 if the action is not allowed in the state
        logger.info("Computing row map")
        row_map = np.zeros(nr_states * nr_actions, dtype=np.int32)
        row_map[:] = -1
        for state in range(nr_states):
            for action_offset in range(pomdp.get_nr_available_actions(state)):
                labels = pomdp.choice_labeling.get_labels_of_choice(pomdp.get_choice_index(state, action_offset))
                for label in labels:
                    if label != cls.NO_LABEL:
                        action_idx = action_labels2indices[label]
                        row_map[state * nr_actions + action_idx] = pomdp.transition_matrix.get_rows_for_group(state)[action_offset]

        # Transitions
        logger.info("Computing transitions")
        transitions = SparseArray.from_data(nr_states, nr_actions, row_map, pomdp.transition_matrix)

        # Allowed actions
        logger.info("Computing allowed actions")
        allowed_actions = (row_map != -1).reshape(nr_states, nr_actions)

        # Sinks
        logger.info("Computing sinks")
        sinks = ~allowed_actions.any(axis=-1)

        # Raw rewards
        logger.info("Computing raw rewards")
        raw_rewards = {}

        for reward_type in rewards_types:
            raw_rewards[reward_type] = SparseArray.zeros_like(transitions)
            reward_model = pomdp.reward_models[reward_type]
            if reward_model.has_state_rewards:
                rewards = np.array(reward_model.state_rewards)
                for sa_idx in range(nr_states * nr_actions):
                    next_states = raw_rewards[reward_type].get_row_indices_np(sa_idx)
                    raw_rewards[reward_type].get_row_np(sa_idx)[:] += rewards[next_states]

            if reward_model.has_transition_rewards:
                raise NotImplementedError("Transition rewards are not supported")

            if reward_model.has_state_action_rewards:
                state_action_rewards = np.array(reward_model.state_action_rewards)
                for sa_idx in range(nr_states * nr_actions):
                    row_idx = row_map[sa_idx]
                    if row_idx == -1:
                        continue
                    raw_rewards[reward_type].get_row_np(sa_idx)[:] += state_action_rewards[row_idx]

        # Assign labels to states
        logger.info("Computing labels")
        labeling = pomdp.labeling
        labels = {}
        for label in labeling.get_labels():
            labels[label] = np.zeros(nr_states, dtype=bool)
            for state in labeling.get_states(label):
                labels[label][state] = 1

        # Rewards
        logger.info("Computing scalarized rewards")
        rewards = SparseArray.zeros_like(transitions)
        reward_data = {
            reward_type: raw_rewards[reward_type].data
            for reward_type in rewards_types
        }
        rewards.data = get_scalarized_reward(reward_data, rewards_types)

        # Metalabels
        logger.info("Computing metalabels")
        metalabel_keys = None if metalabels is None else list(metalabels.keys())
        metalabels_data = None
        if metalabel_keys is not None:
            metalabels_data = np.ones((nr_states, len(metalabels)), dtype=bool)
            for i, m in enumerate(metalabel_keys):
                for l in metalabels[m]:
                    metalabels_data[:, i] &= labels[l]
        else:
            metalabels_data = np.zeros((nr_states, 0), dtype=bool)

        # Observations
        logger.info("Computing observations")
        valuations = pomdp.observation_valuations
        nr_observables = len(json.loads(str(valuations.get_json(0))))

        observations_by_ids = np.zeros((pomdp.nr_observations, nr_observables), dtype=np.float32)
        observation_labels = list(json.loads(str(valuations.get_json(0))).keys())

        for obs_id in range(pomdp.nr_observations):
            valuation_json = json.loads(str(valuations.get_json(obs_id)))
            observations_by_ids[obs_id] = np.array(list(valuation_json.values()), dtype=np.float32)

        state_observation_ids = np.array(pomdp.observations)
        observations = observations_by_ids[state_observation_ids]

        # Save simulator data
        return Simulator(
            id = Simulator.get_free_id(),
            initial_state = initial_state,
            max_outcomes = (transitions.row_ends-transitions.row_starts).max(),
            max_steps = max_steps,
            random_init = random_init,
            transitions = cast2jax(transitions),
            rewards = cast2jax(rewards),
            observations = cast2jax(observations),
            sinks = cast2jax(sinks),
            allowed_actions = cast2jax(allowed_actions),
            labels = labels,
            metalabels = cast2jax(metalabels_data),
            action_labels = action_labels,
            observation_labels = observation_labels,
        )


class StormVecEnv:
    """
        Class that provides a fast sparse vectorized representation of a Storm environment.
        It uses JAX to compile the topology extracted from the given model, thus accelerating the interactions.
    """

    def __init__(self, pomdp: SparsePomdp, get_scalarized_reward, num_envs=1, seed=42, metalabels=None, random_init=False, max_steps=100):
        self.simulator_states = States(
            vertices = jnp.zeros(num_envs, jnp.int32),
            steps = jnp.zeros(num_envs, jnp.int32),
        )
        self.rng_key = jax.random.key(seed)
        self.random_init = random_init
        self.simulator = StormVecEnvBuilder.build_from_pomdp(
            pomdp,
            get_scalarized_reward,
            num_envs=num_envs,
            seed=seed,
            metalabels=metalabels,
            max_steps=max_steps,
            random_init=random_init
        )

    def enable_random_init(self):
        self.simulator.id = Simulator.get_free_id()
        self.simulator.random_init = True
    
    def disable_random_init(self):
        self.simulator.id = Simulator.get_free_id()
        self.simulator.random_init = False
    
    def set_num_envs(self, num_envs):
        self.simulator_states = States(
            vertices = jnp.zeros(num_envs, jnp.int32),
            steps = jnp.zeros(num_envs, jnp.int32),
        )
    
    def set_seed(self, seed):
        self.rng_key = jax.random.key(seed)

    def reset(self):
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        res: ResetInfo = self.simulator.reset(self.simulator_states, reset_key if self.random_init else None)
        self.simulator_states = res.states
        return res.observations, res.allowed_actions, res.metalabels
    
    def step(self, actions):
        self.rng_key, step_key = jax.random.split(self.rng_key)
        res: StepInfo = self.simulator.step(self.simulator_states, actions, step_key)
        self.simulator_states = res.states
        return res.observations, res.rewards, res.done, res.allowed_actions, res.metalabels

    def get_label(self, label, vertices=None):
        if vertices is None:
            return self.get_label(label, self.simulator_states.vertices)
    
        return self.simulator.labels[label][vertices]

    def get_labels(self, vertices=None):
        if vertices is None:
            return self.get_labels(self.simulator_states.vertices)
        
        return {
            key: val[vertices] for key, val in self.simulator.labels.items()
        }

    def get_action_labels(self):
        """
            Get list of action labels that occur in the environment.
        """
        return self.simulator.action_labels

    def get_observation_labels(self):
        """
            Get list of observation labels that occur in the environment.
        """
        return self.simulator.observation_labels

    def save(self, file: str):
        pickle.dump(self, open(file, "wb"))
    
    @classmethod
    def load(cls, file: str):
        return pickle.load(open(file, "rb" ))
