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

### Requirements
- JAX
- NumPy
- Stormpy
- PAYNT (to be removed)

## Usage
The following code snippet demonstrates how to use StormVecEnv:
```python
import os
import stormpy
from vec_storm import StormVecEnv

# Load the Storm model
path_to_model = "path/to/model/dir"  # A directory containing `sketch.templ` and `sketch.props`
sketch_path = os.path.join(env_path, "sketch.templ")
properties_path = os.path.join(env_path, "sketch.props")    
quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
pomdp = quotient.pomdp

# Define the scalar reward function based on the reward signals in the model
scalarize_reward = lambda r: r['reward1'] + 2 * r['reward2']

# Create the StormVecEnv
env = StormVecEnv(pomdp, scalarize_reward, seed=42, num_envs=1, max_steps=100)

# Reset the environment
obs, allowed_actions, _ = env.reset()

# Take a step in the environment
obs, reward, done, truncated, allowed_actions, _ = env.step(numpy.array([0]))
```

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
