import os
import time

import numpy as np

import paynt.parser.sketch

from vec_storm import StormVecEnv


def benchmark(pomdp):
    for l in range(13):
        r = 2000
        n = 2**l

        if n > 1000:
            r //= 2**(l-8)

        env = StormVecEnv(pomdp, get_scalarized_reward, n)

        start = time.time()
        env.reset()
        env.step(np.array([2]*n))
        for _ in range(r):
            env.step(np.array([0]*n))
            env.step(np.array([5]*n))
            env.step(np.array([3]*n))
            env.step(np.array([0]*n))
            env.step(np.array([0]*n))
            env.step(np.array([0]*n))
        

        end = time.time()

        print(f"{n=}", end=":\t")
        print(f"Time: {end-start:.3f}", end=",\t")
        print(f"Steps s^-1: { r / (end-start):.3f}", end=",\t")
        print(f"EnvSteps s^-1: {n * r / (end-start):.3f}")


if __name__ == '__main__':
    def load_pomdp(env_path):
        env_path = os.path.abspath(env_path)
        sketch_path = os.path.join(env_path, "sketch.templ")
        properties_path = os.path.join(env_path, "sketch.props")    
        quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
        return quotient.pomdp
    
    model_path = "/opt/learning/synthesis/rl_src/models/refuel-10"
    pomdp = load_pomdp(model_path)

    def get_scalarized_reward(rewards):
        return 100*rewards["refuels"] + 10*rewards["costs"] + rewards["steps"]

    benchmark(pomdp)
    
    # n = 1
    # env = StormVecEnv(pomdp, get_scalarized_reward, n)
    # def print_res(res):
    #     observations = res[0].T
    #     o = {l: o[:10] for l, o in zip(env.observation_labels, observations)}
    #     print('----\nStep')
    #     print(f"reward = {res[1][0]}")
    #     print(f"done = {res[2][0]}")
    #     print(f"action_mask = {res[3][0]}")
    #     print(f"observations = {o}")
    
    # env.reset()
    # print_res(env.step(np.array([2]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([5]*n)))
    # print_res(env.step(np.array([3]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
    # print_res(env.step(np.array([0]*n)))
