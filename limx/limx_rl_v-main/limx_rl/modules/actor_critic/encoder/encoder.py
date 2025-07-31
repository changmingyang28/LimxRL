from abc import ABC, abstractmethod
from typing import Union
import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    is_detach: bool
    @abstractmethod
    def reset(self, dones=None):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args) -> Union[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def calculate_loss(self, *args) -> torch.Tensor:
        raise NotImplementedError