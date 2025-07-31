from setuptools import find_packages
from distutils.core import setup

setup(
    name='limx_legged_gym',
    version='1.0.0',
    author="LimX Dynamics",
    maintainer="Darrell Shiqin Dai",
    maintainer_email="darrelldai@limxdynamics.com",
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'limx_rl',
                      'matplotlib']
)
