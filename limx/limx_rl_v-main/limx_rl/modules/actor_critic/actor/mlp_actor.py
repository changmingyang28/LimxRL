from typing import Optional

from torch import Tensor

from .actor import Actor
from limx_rl.modules.actor_critic.mlp import MLP


class MlpActor(Actor):
    def __init__(self,
                 num_obs,
                 num_actions,
                 init_noise_std,
                 hidden_dims=[256, 256, 256],
                 activation="elu",
                 orthogonal_init=False,
                 **kwargs):
        if kwargs:
            print(
                "MlpActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(num_obs,
                         num_actions,
                         init_noise_std)
        self.mlp = MLP(num_obs, num_actions, hidden_dims, activation, orthogonal_init)

    def reset(self, dones=None):
        pass
    def forward(self, obs, *args, **kwargs):
        return self.mlp(obs)
