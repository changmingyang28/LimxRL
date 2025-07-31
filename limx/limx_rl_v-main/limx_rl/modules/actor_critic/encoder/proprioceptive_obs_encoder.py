from limx_rl.modules.actor_critic.mlp import MLP
from limx_rl.modules.actor_critic.encoder.encoder import Encoder
import torch


class ProprioceptiveObsEncoder(Encoder):
    def __init__(self, input_dim, output_dim, hidden_dims, is_detach, **kwargs):
        if kwargs:
            print(
                "ProprioceptiveObsEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ProprioceptiveObsEncoder, self).__init__()
        self.is_detach = is_detach
        self.output_dim = output_dim
        self.mlp = MLP(input_dim,
                       output_dim,
                       hidden_dims=hidden_dims,
                       activation="elu",
                       orthogonal_init=False)
    def reset(self, dones=None):
        pass
    def forward(self, proprioceptive_obs_history, is_detach=False):
        encoder_out = self.mlp(proprioceptive_obs_history)
        if is_detach:
            return encoder_out.detach()
        else:
            return encoder_out

    def calculate_loss(self, output, target) -> torch.Tensor:
        return (output - target).pow(2).mean()
