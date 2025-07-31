import os
from isaacgym import gymapi
import numpy as np
from typing import Tuple

# Try to import moviepy, fallback to basic functionality if not available
try:
    import moviepy.editor as mpy
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Video will be saved as individual frames.")


class RecordVideoWrapper:
    def __init__(self, env):
        self._env = env
        # Copy properties from the wrapped environment
        for name in dir(env):
            if isinstance(getattr(type(env), name, None), property):
                if not name.startswith('_'):
                    setattr(self, name, getattr(env, name))
        # Also copy important attributes including gym and all required attributes for RL training
        for attr in ['gym', 'sim', 'envs', 'num_privileged_obs', 'num_obs', 'num_actions', 'num_commands', 'critic_obs_history_length']:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))
        self.reset()
        self._is_recording_video = False
        self._frames_buf = None
        self.fps = None
        self._camera_video_props_are_set = False

    def step(self, actions):
        obs_buf, rew_buf, reset_buf, extras = self._env.step(actions)
        if self._is_recording_video:
            self._frames_buf.append(self._get_camera_image())
        return obs_buf, rew_buf, reset_buf, extras

    def reset(self):
        return self._env.reset()

    def set_camera_video_props(self, frame_size: Tuple[int, int], camera_offset: Tuple[float, float, float],
                               camera_rotation: Tuple[float, float, float], env_idx: int, actor_idx: int,
                               rigid_body_idx: int, fps: int):
        self._env_handle = self._env.envs[env_idx]
        self._camera_properties = gymapi.CameraProperties()
        self._camera_properties.width, self._camera_properties.height = frame_size
        self._camera_handle = self.gym.create_camera_sensor(self._env_handle, self._camera_properties)
        camera_offset = gymapi.Vec3(*camera_offset)
        camera_rotation = gymapi.Quat.from_euler_zyx(*np.deg2rad(camera_rotation))
        self._actor_handle = self.gym.get_actor_handle(self._env_handle, actor_idx)
        self._body_handle = self.gym.get_actor_rigid_body_handle(self._env_handle, self._actor_handle, rigid_body_idx)
        self.gym.attach_camera_to_body(self._camera_handle, self._env_handle, self._body_handle,
                                       gymapi.Transform(camera_offset, camera_rotation),
                                       gymapi.FOLLOW_POSITION)
        self.fps = fps
        self._camera_video_props_are_set = True

    def get_observations(self):
        return self._env.get_observations()
    
    def __getattr__(self, name):
        # Forward any missing attributes to the wrapped environment
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._env, name)

    def _get_camera_image(self):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        image = self.gym.get_camera_image(self.sim, self._env_handle, self._camera_handle,
                                          gymapi.IMAGE_COLOR)
        image = image.reshape((self._camera_properties.height, self._camera_properties.width, 4))
        return image

    @property
    def is_recording_video(self):
        return self._is_recording_video

    def start_recording_video(self):
        if not self._camera_video_props_are_set:
            raise RuntimeError("Camera properties are not set!")
        elif self._is_recording_video:
            raise RuntimeError("Videos are already recording! It should be ended first before starting recording.")
        else:
            self._is_recording_video = True
            if self._frames_buf is None:
                self._frames_buf = []

    def end_and_save_recording_video(self, video_path, filename):
        if not os.path.isdir(video_path):
            os.makedirs(video_path, exist_ok=True)
        
        if MOVIEPY_AVAILABLE and len(self._frames_buf) > 0:
            # Save as MP4 video
            clip = mpy.ImageSequenceClip(self._frames_buf, fps=self.fps)
            self._frames_buf = []
            self._is_recording_video = False
            clip.write_videofile(os.path.join(video_path, filename), codec="libx264")
        else:
            # Save as individual PNG frames
            frame_count = len(self._frames_buf)
            for i, frame in enumerate(self._frames_buf):
                frame_filename = f"{filename}_frame_{i:04d}.png"
                from PIL import Image
                # Convert RGBA to RGB
                rgb_frame = frame[:, :, :3]
                img = Image.fromarray(rgb_frame.astype(np.uint8))
                img.save(os.path.join(video_path, frame_filename))
            
            self._frames_buf = []
            self._is_recording_video = False
            print(f"Saved {frame_count} frames to {video_path}")

    @property
    def frames_buf(self):
        return self._frames_buf