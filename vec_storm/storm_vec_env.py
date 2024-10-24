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


class StormVecEnv:
    """
        Class that provides a fast sparse vectorized representation of a Storm environment.
        It uses JAX to compile the topology extracted from the given model, thus accelerating the interactions.
    """

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

    def __init__(self, pomdp: SparsePomdp, get_scalarized_reward, num_envs=1, seed=42, metalabels=None, random_init=False, max_steps=100):
        """
            pomdp: The POMDP object that should be compiled into a jax-based environment.
            get_scalarized_reward: A function that accepts a dictionary indexed by reward signal names and returns a number.
        """
        self.simulator_states = States(
            vertices = jnp.zeros(num_envs, jnp.int32),
            steps = jnp.zeros(num_envs, jnp.int32),
        )
        self.rng_key = jax.random.key(seed)

        sim = simulator.create_simulator(pomdp)
        self.action_labels = self.get_action_labels(pomdp, self.NO_LABEL)
        self.action_labels2indices = {label: i for i, label in enumerate(self.action_labels)}
        self.initial_state = pomdp.initial_states[0]
        self.random_init = random_init

        rewards_types = sim.get_reward_names()
        nr_states = pomdp.nr_states
        nr_actions = len(self.action_labels)

        # Row map: Assigns each (state, action) pair a row in the spare transition/reward matrix
        # or -1 if the action is not allowed in the state
        row_map = np.zeros(nr_states * nr_actions, dtype=np.int32)
        row_map[:] = -1
        for state in range(nr_states):
            for action_offset in range(pomdp.get_nr_available_actions(state)):
                labels = pomdp.choice_labeling.get_labels_of_choice(pomdp.get_choice_index(state, action_offset))
                for label in labels:
                    if label != self.NO_LABEL:
                        action_idx = self.action_labels2indices[label]
                        row_map[state * nr_actions + action_idx] = pomdp.transition_matrix.get_rows_for_group(state)[action_offset]

        # Transitions
        self.transitions = SparseArray.from_data(nr_states, nr_actions, row_map, pomdp.transition_matrix)

        # Allowed actions
        self.allowed_actions = (row_map != -1).reshape(nr_states, nr_actions)

        # Sinks
        self.sinks = ~self.allowed_actions.any(axis=-1)

        # Raw rewards
        raw_rewards = {}

        for reward_type in rewards_types:
            raw_rewards[reward_type] = SparseArray.zeros_like(self.transitions)
            reward_model = pomdp.reward_models[reward_type]
            if reward_model.has_state_rewards:
                rewards = np.array(reward_model.state_rewards)
                for sa_idx in range(nr_states * nr_actions):
                    next_states = raw_rewards[reward_type].get_row_indices_np(sa_idx)
                    raw_rewards[reward_type].get_row_np(sa_idx)[:] += rewards[next_states]

            if reward_model.has_transition_rewards:
                raise NotImplementedError("Transition rewards are not supported")

            if reward_model.has_state_action_rewards:
                for sa_idx in range(nr_states * nr_actions):
                    row_idx = row_map[sa_idx]
                    if row_idx == -1:
                        continue
                    raw_rewards[reward_type].get_row_np(sa_idx)[:] += reward_model.state_action_rewards[row_idx]

        # Assign labels to states
        labeling = pomdp.labeling
        self.labels = {}
        for label in labeling.get_labels():
            self.labels[label] = np.zeros(nr_states, dtype=bool)
            for state in labeling.get_states(label):
                self.labels[label][state] = 1

        # Rewards
        self.rewards = SparseArray.zeros_like(self.transitions)
        reward_data = {
            reward_type: raw_rewards[reward_type].data
            for reward_type in rewards_types
        }
        self.rewards.data = get_scalarized_reward(reward_data, rewards_types)

        # Metalabels
        self.metalabels = None if metalabels is None else list(metalabels.keys())
        self.metalabels_data = None
        if self.metalabels is not None:
            self.metalabels_data = np.ones((nr_states, len(metalabels)), dtype=bool)
            for i, m in enumerate(self.metalabels):
                for l in metalabels[m]:
                    self.metalabels_data[:, i] &= self.labels[l]
        else:
            self.metalabels_data = np.zeros((nr_states, 0), dtype=bool)

        # Observations
        valuations = pomdp.observation_valuations
        observations = pomdp.observations
        nr_observables = len(json.loads(str(valuations.get_json(0))))

        self.observations = np.zeros((nr_states, nr_observables), dtype=np.float32)
        self.observation_labels = list(json.loads(str(valuations.get_json(0))).keys())
        for state in range(nr_states):
            observation_id = observations[state]
            valuation_json = json.loads(str(valuations.get_json(observation_id)))
            self.observations[state] = np.array(list(valuation_json.values()), dtype=np.float32)

        # Save simulator data
        self.simulator = Simulator(
            initial_state = self.initial_state,
            max_outcomes = (self.transitions.row_ends-self.transitions.row_starts).max(),
            max_steps = max_steps,
            transitions = cast2jax(self.transitions),
            rewards = cast2jax(self.rewards),
            observations = cast2jax(self.observations),
            sinks = cast2jax(self.sinks),
            allowed_actions = cast2jax(self.allowed_actions),
            metalabels = cast2jax(self.metalabels_data),
        )

    def enable_random_init(self):
        self.random_init = True
    
    def disable_random_init(self):
        self.random_init = False
    
    def set_seed(self, seed):
        self.rng_key = jax.random.key(seed)

    def reset(self):
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        res: ResetInfo = self.simulator.reset(self.simulator_states, reset_key if self.random_init else None)
        self.simulator_states = res.states
        return res.observations, res.allowed_actions, res.metalabels
    
    def step(self, actions):
        self.rng_key, step_key = jax.random.split(self.rng_key)
        res: StepInfo = self.simulator.step(self.simulator_states, actions, step_key, random_init=self.random_init)
        self.simulator_states = res.states
        return res.observations, res.rewards, res.done, res.allowed_actions, res.metalabels

    def get_label(self, label, vertices=None):
        if vertices is None:
            return self.get_label(label, self.simulator_states.vertices)
    
        return self.labels[label][vertices]

    def get_labels(self, vertices=None):
        if vertices is None:
            return self.get_labels(self.simulator_states.vertices)
        
        return {
            key: val[vertices] for key, val in self.labels.items()
        }

    def save(self, file: str):
        pickle.dump(self, open(file, "wb"))
    
    @classmethod
    def load(cls, file: str):
        return pickle.load(open(file, "rb" ))
