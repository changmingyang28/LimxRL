from .actor import Actor
from limx_rl.modules.actor_critic.mlp import MLP
import torch
import torch.optim as optim


class EncodedMlpActor(Actor):
    def __init__(self,
                 encoder,
                 num_obs,
                 num_actions,
                 hidden_dims=[256, 256, 256],
                 activation="elu",
                 orthogonal_init=False,
                 **kwargs):
        if kwargs:
            print(
                "EncodedMlpActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(num_obs, num_actions)
        self.is_recurrent = getattr(encoder, "is_recurrent", False)
        self.encoder = encoder
        self.mlp = MLP(encoder.output_dim, num_actions, hidden_dims, activation, orthogonal_init)
        if encoder is None:
            raise ValueError("Encoder is not defined")

    def reset(self, dones):
        self.encoder.reset(dones=dones)

    def forward(self, observations, obs_history, *args, masks=None, hidden_states=None, **kwargs):
        encoder_output = self.encoder(input=observations, input_history=obs_history, masks=masks,
                                      hidden_states=hidden_states, is_detach=self.encoder.is_detach)
        actor_output = self.mlp(encoder_output)
        return actor_output

    def get_hidden_states(self):
        if self.is_recurrent:
            return self.encoder.get_hidden_states()
        else:
            pass

    def get_parameters(self):
        return list(self.encoder.parameters()), list(self.mlp.parameters())
