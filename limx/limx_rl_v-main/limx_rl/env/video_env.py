from limx_rl.env.vec_env import VecEnv
import torch
from abc import abstractmethod
from typing import Tuple, List


class VideoEnv(VecEnv):
    frame_buf: List[torch.tensor]
    is_recording_video: bool

    @abstractmethod
    def set_camera_video_props(self, frame_size: Tuple[int, int], camera_offset: Tuple[float, float, float],
                               camera_rotation: Tuple[float, float, float], env_idx: int, actor_idx: int,
                               rigid_body_idx: int, fps: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_recording_video(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_and_save_recording_video(self, video_path, filename) -> None:
        raise NotImplementedError
