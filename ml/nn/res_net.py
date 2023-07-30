
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.data import Data
from .model import Model


class ResNet(Model):
    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 n_embed=32,
                 n_blocks=1,
                 activation=None,
                 class_weights=None,
                 name="ResNet",
                 embed_nan=False,
                 **kwargs):
        super().__init__()

        self.n_embed = n_embed
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_blocks = n_blocks
        self.name = name

        self.activation = nn.GELU(approximate="tanh") if activation is None else activation

        layers = [nn.Linear(n_features, n_embed, bias=False)]
        for _ in range(n_blocks):
            layers.append(nn.LayerNorm(n_embed))
            layers.append(nn.Linear(n_embed, n_embed, bias=False))

        layers.append(nn.LayerNorm(n_embed))
        layers.append(nn.Linear(n_embed, n_classes))

        self.layers = nn.ModuleList(layers)

        self.w_nan = nn.Parameter(torch.randn(n_features)) if embed_nan else None

        self.class_weights = nn.Parameter(class_weights, requires_grad=False) if class_weights is not None else None

    def forward(self, batch: Data, return_loss=False, return_all=False):
        x_continuous = batch.x_continuous
        x_categorical = batch.x_categorical
        x = torch.cat([x_continuous, x_categorical], dim=1)

        w_nan = self.w_nan if self.w_nan is not None else 0
        x = torch.where(torch.isnan(x), w_nan, x)

        x = self.layers[0](x)
        i = 1
        while i < len(self.layers) - 2:
            ln = self.layers[i]
            linear = self.layers[i + 1]

            x = x + linear(ln(self.activation(x)))

            i += 2

        ln = self.layers[i]
        linear = self.layers[i + 1]

        logits = linear(ln(self.activation(x)))

        loss = None
        if return_loss or return_all:
            y = batch.y
            loss = F.cross_entropy(logits, y, weight=self.class_weights)

        if return_all:
            return logits, loss
        elif return_loss:
            return loss
        return logits

    @property
    def hyperparameters(self):
        return {
            "n_embed": self.n_embed,
            "n_classes": self.n_classes,
            "n_blocks": self.n_blocks,
            "n_parameters": self.n_params,
            "w_nan": self.w_nan is not None,
        }

    @property
    def estimated_initial_loss(self):
        return -torch.tensor(1 / self.n_classes).log()

    def predict(self, batch, available_classes=None):
        logits = self(batch)

        if available_classes is not None:
            forbidden_classes = torch.tensor([i for i in range(self.n_classes) if i not in available_classes])
            logits[:, forbidden_classes] = -torch.inf

        return torch.argmax(logits, dim=1)