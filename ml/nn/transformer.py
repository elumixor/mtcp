from __future__ import annotations

import torch
import torch.nn as nn

from .model import Model
from .my_embed import MyEmbed


class Transformer(Model):
    def __init__(
        self,
        n_features_continuous: int,
        categorical_sizes: list[int],
        n_classes: int,
        n_embed=32,
        n_inner=None,
        n_blocks=1,
        n_heads=1,
        activation=None,
        dropout=0.3,
        class_weights=None,
        name="transformer",
        **kwargs
    ):
        super().__init__()

        if n_inner is None:
            n_inner = n_embed * 4

        self.n_embed = n_embed
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_inner = n_inner
        self.dropout = dropout
        self.n_classes = n_classes
        self.name = name
        self.class_weights = (
            nn.Parameter(class_weights, requires_grad=False)
            if class_weights is not None
            else None
        )

        self.activation = (
            nn.GELU(approximate="tanh") if activation is None else activation
        )
        self.embed = MyEmbed(n_features_continuous, categorical_sizes, n_embed)
        self.blocks = (
            nn.Sequential(
                *[
                    Block(
                        n_embed,
                        activation=self.activation,
                        n_inner=n_inner,
                        n_heads=n_heads,
                        dropout=dropout,
                    )
                    for _ in range(n_blocks)
                ]
            )
            if n_blocks > 0
            else nn.Identity()
        )
        self.ln = nn.LayerNorm(n_embed)
        self.logits = nn.Linear(n_embed, n_classes)

    def get_logits(self, x_continuous, x_categorical):
        x = self.embed(x_continuous, x_categorical)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.logits(x[:, 0, :])
        return logits

    @property
    def hyperparameters(self):
        return {
            "n_embed": self.n_embed,
            "n_inner": self.n_inner,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "n_classes": self.n_classes,
            "n_parameters": self.n_params,
            "dropout": self.dropout,
        }

    @property
    def estimated_initial_loss(self):
        return -torch.tensor(1 / self.n_classes).log()


class Block(nn.Module):
    def __init__(
        self, n_embed: int, n_inner=None, n_heads=1, activation=None, dropout=0.0
    ):
        super().__init__()

        if n_inner is None:
            n_inner = n_embed * 4

        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.attention = Attention(n_embed, n_heads=n_heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(n_embed, n_inner),  # bias is already in the LayerNorm
            activation,
            nn.Linear(n_inner, n_embed),
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class Attention(nn.Module):
    """
    Classic attention module with multi-head support
    """

    def __init__(self, n_embed: int, n_heads=1, dropout=0.0, bias=False):
        super().__init__()

        assert (
            torch.__version__ >= "2.0.0"
        ), "MaskedAttentiveEmbed requires PyTorch 2.0.0 or higher"
        assert (
            n_embed % n_heads == 0
        ), f"Embedding size ({n_embed}) must be divisible by the number of heads ({n_heads})"

        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        self.dropout = dropout

        self.linear = nn.Linear(
            n_embed, 3 * n_embed, bias=bias
        )  # 3 is to have a single operation for q, k, v

        self.residual_dropout = nn.Dropout(dropout)
        self.residual_project = nn.Linear(n_embed, n_embed, bias=bias)

    def forward(self, x):
        B, T, E = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.linear(x).split(E, dim=2)

        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, E)
        )  # re-assemble all head outputs side by side

        y = self.residual_dropout(self.residual_project(y))

        return y
