#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages
from distutils.core import setup

setup(
    name="limx_rl",
    version="1.0.0",
    packages=find_packages(),
    author="LimX Dynamics",
    maintainer="Darrell Shiqin Dai",
    maintainer_email="darrelldai@limxdynamics.com",
    # url="https://github.com/leggedrobotics/limx_rl",
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "GitPython",
        "onnx",
    ],
)
