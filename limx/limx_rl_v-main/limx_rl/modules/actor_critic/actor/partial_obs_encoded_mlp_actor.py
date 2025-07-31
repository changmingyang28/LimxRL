from .actor import Actor
from limx_rl.modules.actor_critic.mlp import MLP
import torch


class ParitialObsEncodedMlpActor(Actor):
    def __init__(self,
                 encoder,
                 num_obs,
                 num_commands,
                 num_actions,
                 hidden_dims=[256, 256, 256],
                 activation="elu",
                 orthogonal_init=False,
                 **kwargs):
        if kwargs:
            print(
                "ParitialObsEncodedMlpActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(num_obs, num_commands, num_actions)
        self.encoder = encoder
        self.mlp = MLP(num_obs + num_commands + encoder.output_dim, num_actions, hidden_dims, activation,
                       orthogonal_init)
        if encoder is None:
            raise ValueError("Encoder is not defined")

    def reset(self, dones):
        self.encoder.reset(dones=dones)

    def forward(self, observations, obs_history, *args, **kwargs):
        encoder_output = self.encoder(obs_history, self.encoder.is_detach)
        observations_with_encoder_output = torch.cat((observations, encoder_output), dim=-1)
        actor_output = self.mlp(observations_with_encoder_output)
        return actor_output

    def get_parameters(self):
        return list(self.encoder.parameters()), list(self.mlp.parameters())