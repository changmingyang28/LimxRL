from limx_rl.utils.utils import get_activation
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(
            self,
            num_input_dim,
            num_output_dim,
            hidden_dims=[256, 256],
            activation="elu",
            orthogonal_init=False,
    ):
        super(MLP, self).__init__()
        self.orthogonal_init = orthogonal_init
        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim

        activation = get_activation(activation)

        # MLP
        mlp_layers = []
        mlp_layers.append(nn.Linear(num_input_dim, hidden_dims[0]))
        if self.orthogonal_init:
            nn.init.orthogonal_(mlp_layers[-1].weight, np.sqrt(2))
        mlp_layers.append(activation)
        for layer in range(len(hidden_dims)):
            if layer == len(hidden_dims) - 1:
                mlp_layers.append(nn.Linear(hidden_dims[layer], num_output_dim))
                if self.orthogonal_init:
                    nn.init.orthogonal_(mlp_layers[-1].weight, 0.01)
                    nn.init.constant_(mlp_layers[-1].bias, 0.0)
            else:
                mlp_layers.append(nn.Linear(hidden_dims[layer], hidden_dims[layer + 1]))
                if self.orthogonal_init:
                    nn.init.orthogonal_(mlp_layers[-1].weight, np.sqrt(2))
                    nn.init.constant_(mlp_layers[-1].bias, 0.0)
                mlp_layers.append(activation)
        self.sequential = nn.Sequential(*mlp_layers)

    def forward(self, input):
        return self.sequential(input)
