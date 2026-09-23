import os
import paynt.parser.sketch
from pathlib import Path




AVOID_DET = Path(os.path.abspath(__file__)).parent / 'models/det_avoid'
AVOID_RAND = Path(os.path.abspath(__file__)).parent / 'models/rand_avoid'

def load_pomdp(env_path):
    env_path = os.path.abspath(env_path)
    sketch_path = os.path.join(env_path, "sketch.templ")
    properties_path = os.path.join(env_path, "sketch.props")    
    quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    print(quotient.pomdp)
    return quotient.pomdp


def generate_reward_selection_function(rewards, labels):
    if labels is None or len(labels) == 0:
        return -1.0
    last_reward = labels[-1]
    return rewards[last_reward]


def obs_to_dict(env, obs):
    return {
        key: val for key, val in zip(env.get_observation_labels(), obs)
    }