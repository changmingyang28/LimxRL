from .critic import Critic
from limx_rl.modules.actor_critic.mlp import MLP


class MlpCritic(Critic):
    def __init__(self, num_critic_obs,
                 num_commands,
                 hidden_dims=[256, 256, 256],
                 activation="elu",
                 orthogonal_init=False,
                 **kwargs):
        if kwargs:
            print(
                "MlpActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(num_critic_obs, num_commands)
        self.mlp = MLP(num_critic_obs + num_commands, 1, hidden_dims, activation, orthogonal_init)

    def reset(self, dones=None):
        pass

    def forward(self, critic_obs, *args, **kwargs):
        return self.mlp(critic_obs)

    def evaluate(self, critic_observations, *args, **kwargs):
        value = self(critic_observations)
        return value
