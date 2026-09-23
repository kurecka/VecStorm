from setuptools import setup, find_packages

setup(
    name="VecStorm",
    version="2026.9.1",
    packages=find_packages(),
    install_requires=[
        "jax",
        "chex",
        "numpy",
        "stormpy",
    ],
    extras_require={
        # Only needed to build a POMDP from a PRISM sketch (`paynt.parser.sketch`), e.g.
        # in the tests, demo.py and benchmark.py. The core `vec_storm` package accepts
        # any stormpy POMDP object and never imports paynt itself.
        "sketch": ["paynt"],
        # Only needed for the optional gymnasium-compatible wrappers in
        # `vec_storm.gymnasium_env` (StormGymVecEnv / StormGymEnv).
        "gym": ["gymnasium"],
    },
    author="Martin Kurecka & David Hudak",
    description="A jax based compiler for Storm environments.",
)
