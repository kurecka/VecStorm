from setuptools import setup, find_packages

setup(
    name="VecStorm",
    version="2024.10.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "chex",
        "numpy",
        "stormpy"
    ],
    author="Martin Kurecka",
    description="A jax based compiler for Storm environments.",
)
