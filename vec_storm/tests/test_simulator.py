import os
import tempfile
import paynt.parser.sketch
import numpy as np

from pytest import approx

from vec_storm import StormVecEnv


def load_pomdp(env_path):
    env_path = os.path.abspath(env_path)
    sketch_path = os.path.join(env_path, "sketch.templ")
    properties_path = os.path.join(env_path, "sketch.props")    
    quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return quotient.pomdp


def test_save_load():
    model_path = 'models/refuel-det'
    pomdp = load_pomdp(model_path)

    def get_scalarized_reward(rewards, rewards_types):
        return (rewards["refuels"] * 0) + 42

    env = StormVecEnv(pomdp, get_scalarized_reward, num_envs=1)

    with tempfile.NamedTemporaryFile() as tmp:
        env.save(tmp.name)
        env2 = StormVecEnv.load(tmp.name)
    
    env2.reset()
    res = env2.step(np.array([0]))
    assert res[1] == approx(42)
