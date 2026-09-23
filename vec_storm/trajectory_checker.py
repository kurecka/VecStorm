# Checks if the trajectory is possible in a given POMDP.


from vec_storm.storm_vec_env import StormVecEnv

import numpy as np

from vec_storm.tests.test_tools import *


class TrajectoryChecker:
    def __init__(self, storm_vec_env: StormVecEnv):
        self.storm_vec_env = storm_vec_env

    def get_new_possible_states(self, state):
        transitions = self.storm_vec_env.simulator.transitions
        allowed_actions = self.storm_vec_env.simulator.allowed_actions[state]
        true_allowed_actions = np.where(allowed_actions == 1)[0]
        possible_next_states = []
        for action in true_allowed_actions.tolist():
            next_states = transitions.get_row_indices_np(state, action)
            possible_next_states.extend(next_states)
        return np.array(possible_next_states)

    def check_trajectory(self, obs_sequence):
        # checks if the given observation sequence is possible in the POMDP given any policy using belief supports
        state_to_observation = self.storm_vec_env.simulator.state_observation_ids
        # Convert observation sequence to a sequence of sets of possible states
        possible_states_sequence = []
        for obs in obs_sequence:
            possible_states = np.where(state_to_observation == obs)[0]
            if len(possible_states) == 0:
                raise ValueError(f"No state corresponds to observation {obs}.")
            possible_states_sequence.append(possible_states.tolist())
        belief_support_n = possible_states_sequence[0]

        for t in range(len(obs_sequence) - 1):
            print(f"Len of belief support at time {t}: {len(belief_support_n)}")
            belief_support_n_plus_1 = set()
            for state in belief_support_n:
                possible_next_states = self.get_new_possible_states(state)
                if len(possible_next_states) == 0:
                    continue
                for next_state in possible_next_states:
                    if next_state in possible_states_sequence[t + 1]:
                        belief_support_n_plus_1.add(next_state)
            belief_support_n_plus_1 = belief_support_n_plus_1
            if len(belief_support_n_plus_1) == 0:
                return False
            belief_support_n = belief_support_n_plus_1
        return True


