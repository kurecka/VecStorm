VecStorm is a JAX-based compiler for Storm environments. It takes a Storm model,
generates all possible interactions and stores everything in a sparse manner.

## Installation
Enter the root directory of the repository and run either of the following commands:
```bash
pip install .
pip install -e .
```
The second command installs the package in editable mode, which means that you can
edit the source code and the changes will be reflected in the installed package.

This is also how `VecStorm` is meant to be consumed by other projects: clone/vendor
this repository next to your project and `pip install -e ./VecStorm`, instead of
copying the `vec_storm` sources into your own codebase.

### Requirements
- JAX
- NumPy
- Stormpy

Two extra dependency groups are available for optional functionality that the core
`vec_storm` package itself never imports:
```bash
pip install ".[sketch]"  # PAYNT, only needed to build a POMDP from a PRISM sketch (see demo.py)
pip install ".[gym]"     # gymnasium, only needed for vec_storm.gymnasium_env (see demo_gymnasium.py)
pip install ".[sketch,gym]"
```
Since `StormVecEnv` only needs a stormpy POMDP object (however it was constructed), and
loading a previously saved environment (`StormVecEnv.load`) doesn't even need stormpy,
neither PAYNT nor gymnasium are required to install or use the package itself.

## Usage
The following code snippet demonstrates how to use StormVecEnv. See `demo.py` for a
complete, runnable version (with a random policy over a small vectorized batch of
environments), modelled after how `rl_synthesis`'s `EnvironmentWrapperVec` drives
`StormVecEnv`.
```python
import os
import numpy
import paynt.parser.sketch
from vec_storm import StormVecEnv

# Load the Storm model
path_to_model = "path/to/model/dir"  # A directory containing `sketch.templ` and `sketch.props`
sketch_path = os.path.join(path_to_model, "sketch.templ")
properties_path = os.path.join(path_to_model, "sketch.props")
quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
pomdp = quotient.pomdp

# Define the scalar reward function based on the reward signals in the model
scalarize_reward = lambda r, reward_types: r['reward1'] + 2 * r['reward2']

# Create the StormVecEnv
env = StormVecEnv(pomdp, scalarize_reward, seed=42, num_envs=1, max_steps=100)

# Reset the environment
obs, allowed_actions, _ = env.reset()

# Take a step in the environment
obs, reward, done, truncated, allowed_actions, _ = env.step(numpy.array([0]))
```

Loading a POMDP through PAYNT's sketch parser (as above) is only one way to obtain a
`pomdp` object; `StormVecEnv` itself accepts any stormpy POMDP, however it was built
(e.g. loaded directly from a `.drn` file with plain stormpy).

### Metalabels
The `StormVecEnv` class supports defining precompute conjunctions of atomic labels, called metalabels.
This can be done by passing a dictionary of metalabels to the `metalabels` parameter of the constructor.
```python
metalabels = {
    "metalabel1": ["label1", "label2"],
    "metalabel2": ["label3", "label4"]
}
env = StormVecEnv(pomdp, scalarize_reward, seed=42, num_envs=1, max_steps=100, metalabels=metalabels)

obs, allowed_actions, metalabels = env.reset()
obs, reward, done, truncated, allowed_actions, metalabels = env.step(numpy.array([0]))

# Shape of metalabels: (num_envs, num_metalabels)
# dtype of metalabels: bool
```

Reaching a state where *any* metalabel is set also ends the episode: `done` is `True`
whenever the new state is a POMDP sink, the step limit was reached, **or** any
metalabel is true for that state - even if the underlying model state is not itself a
sink. This lets you mark non-absorbing states (e.g. a goal region on a model that
doesn't declare it as a sink) as terminal purely through metalabels, without changing
the model.

### Just-in-time Compilation
The `StormVecEnv` class uses the `stormpy` library to load a Storm environment to a sparse matrix representation.
The sparse matrix representation is then used in a JAX-based simulator to simulate the environment.
The simulator functions `reset` and `step` are just-in-time compiled using JAX's `jit` decorator for performance.

The simulator is a static parameter in the JITted functions, which means that the function is
compiled for every simulator instance. This allows for more aggressive optimizations by the JAX compiler.
Importantly, different simulator instances need to be distinguished by different values of `id`.
For this reason, it is necessary to use predefined methods like `StormVecEnv()`, `StormVecEnv.load()`, `env.enable_random_init()`, and `env.disable_random_init()` to create and update the simulator.

### Saving and Loading
The `StormVecEnv` class supports saving and loading the simulator which can be useful for large models.
```python
env.save("env.pkl")
env = StormVecEnv.load("env.pkl")
```
It is important to use the prepared interface for saving and loading the simulator!
Otherwise, the JITted functions will not work correctly and an old version of the simulator will be used
in the JITted functions.

## Examples
- `demo.py` - drives `StormVecEnv` directly through its `reset()`/`step()` tuple API,
  the same way `rl_synthesis`'s `EnvironmentWrapperVec` does, but with plain numpy so it
  has no TensorFlow/TF-Agents dependency. Run with `python demo.py`.
- `demo_gymnasium.py` - the same model driven through the optional Gymnasium-compatible
  wrappers described below. Run with `python demo_gymnasium.py` (requires `pip install ".[gym]"`).

## Gymnasium compatibility
`vec_storm.gymnasium_env` (optional, requires `pip install ".[gym]"`) adapts `StormVecEnv`
to the [Gymnasium](https://gymnasium.farama.org/) API:

- `StormGymVecEnv` implements `gymnasium.vector.VectorEnv`. Since `StormVecEnv` already
  runs `num_envs` copies of the environment in lockstep, this is the natural fit and
  requires no extra bookkeeping: VecStorm resets a sub-environment on the step *after*
  it ends, which matches Gymnasium's default `"next-step"` autoreset convention.
- `StormGymEnv` is a `num_envs=1` convenience wrapper implementing the plain
  `gymnasium.Env` API, for tooling that expects a single, non-vectorized environment.

Both expose a `Dict` observation space with an `"observation"` entry (VecStorm's
valuation-based float observation) and an `"action_mask"` entry (the allowed-actions
mask), and a `Discrete`/`MultiDiscrete` action space. `terminated` is `done` with the
truncation reported separately, i.e. `terminated = done & ~truncated`.
```python
from vec_storm import StormVecEnv
from vec_storm.gymnasium_env import StormGymVecEnv

storm_env = StormVecEnv(pomdp, scalarize_reward, num_envs=8, max_steps=100, metalabels=metalabels)
env = StormGymVecEnv(storm_env)

obs, info = env.reset(seed=0)
obs, rewards, terminated, truncated, info = env.step(env.action_space.sample())
```
See `demo_gymnasium.py` for a full runnable example, including the single-environment
`StormGymEnv` variant.
