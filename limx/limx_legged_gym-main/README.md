# Limx Legged Gym: An Isaac Gym Environment for Reinforcement Learning on Legged Robots  #

Welcome to the official repository for the PointFoot Robot project by LimX Dynamics, 
designed for advanced robotic training in locomotion tasks using NVIDIA's Isaac Gym. 
This platform focuses on the seamless transition from sim2real for legged robots.

---

### Installation ###
1. Create a new conda env with python 3.8
    - `conda create -n limx_legged_gym python=3.8`
    - `conda activate limx_legged_gym`
2. Install pytorch 
   - Run `nvidia-smi`, install Pytorch 2.2.1 based on your CUDA version:
      ```
      # CUDA > 11.8
      conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
      # CUDA >= 12.1
      conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
      ```
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Adapt to latest numpy`sed -i 's/np.float/float/' isaacgym/torch_utils.py`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`
4. Install limx_rl (PPO implementation)
    - `git clone ssh://git@192.168.2.65:8022/perception_pr/reinforcement_learning/limx_rl.git`
    -  `cd limx_rl && pip install -e .`
5. Install limx_legged_gym
    - Clone this repository
   - `cd limx_legged_gym && pip install -e .`
6. Install moviepy
   - `pip install moviepy`
7. Install tensorboard
   - `pip install tensorboard`
7. Install wandb (optional)
   - `pip install wandb`


### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one containing  all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

### Usage ###
1. Train:  
  ```python limx_legged_gym/scripts/train.py --task pointfoot_rough```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `pointfoot_rough/logs/<experiment_name>/<run_name>/checkpoints/model_<iteration>.pt`. Where `<experiment_name>` are defined in the train config, and `<run_name>` can be either defined in the train config or generated from date-time. 
    -  The following command line arguments override the values set in the config files:
     - `--task TASK`: Task name.
     - `--num_envs NUM_ENVS`: Number of environments to create. Overrides config file if provided.
     - `--resume`: Resume training from a checkpoint
     - `--experiment_name EXPERIMENT_NAME`: Name of the experiment to run or load.
     - `--run_name RUN_NAME`:  Name of the run.
     - `--load_run LOAD_RUN`:   Name of the run to load when resume=True. If -1: will load the last run.
     - `--checkpoint CHECKPOINT`:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - `--num_envs NUM_ENVS`:  Number of environments to create.
     - `--seed SEED`:  Random seed.
     - `--max_iterations MAX_ITERATIONS`:  Maximum number of training iterations.
     - `--logger LOGGER`: Logger module to use. Choice: `tensorboard`, `wandb`, `neptune`
     - `--video`: Record video during training. Headless mode also works.
     - `--record_length RECORD_LENGTH`: The number of steps to record for videos.
     - `--record_interval RECORD_INTERVAL`: The number of steps to record for videos.
     - `--fps FPS`: The fps of recorded videos.
     - `--frame_size FRAME_SIZE`: The size of recorded frame.
     - `--camera_offset CAMERA_OFFSET`: The offset of the video filming camera.
     - `--camera_rotation CAMREA_ROTATION`: The rotation of the video filming camera.
     - `--env_idx_record ENV_IDX_RECORD`: The env idx to record.
     - `--actor_idx_record ACTOR_IDX_RECORD`: The actor idx to record.
     - `--rigid_body_idx_record RIGID_BODY_IDX_RECORD`: The rigid_body idx to record.
2. Play a trained policy:  
```python limx_legged_gym/scripts/play.py --task=pointfoot_rough --experiment_name pointfoot_rough_recurrent --load_run -1```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and has no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resources/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `limx_legged_gym/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!


