#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Submodule defining the environment definitions."""

from .vec_env import VecEnv
from .video_env import VideoEnv

__all__ = ["VecEnv"]
