# LIMX RL

Fast and simple implementation of RL algorithms, designed to run fully on GPU.
This code is an evolution of `rl-pytorch` provided with NVIDIA's Isaac GYM.

Only PPO is implemented for now. More algorithms will be added later.
Contributions are welcome.

# Features
* Modularized Actor-critic: actor and critic can be customized with any network architecture and forward flow, e.g. MLP linear velocity encoder.
* Video Recorder: record training videos when used with Limx Legged Gym.


## Setup

Following are the instructions to setup the repository for your workspace:

```bash
git clone ssh://git@192.168.2.65:8022/perception_pr/reinforcement_learning/limx_rl.git
cd limx_rl
pip install -e .
```

The framework supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/


### Useful Links

Environment repositories using the framework:

* `Limx-Legged-Gym` (built on top of Legged-Gym): ssh://git@192.168.2.65:8022/perception_pr/reinforcement_learning/limx_legged_gym.git
