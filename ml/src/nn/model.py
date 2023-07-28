from abc import ABC, abstractmethod
import torch.nn as nn

from src.data import Data


class Model(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: Data, return_loss=False, return_all=False):
        pass

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
