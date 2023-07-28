import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import Data

from .model import Model


class Transformer(Model):
    def __init__(self,
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
                 name="transformer"):
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
        self.class_weights = class_weights

        self.nn = TransformerNN(
            n_features_continuous, 
            categorical_sizes,
            n_classes,
            n_embed,
            n_inner,
            n_blocks,
            n_heads,
            activation,
            dropout,
        )

    def forward(self, batch: Data, return_loss=False, return_all=False):
        logits = self.nn(batch.x_continuous, batch.x_categorical)

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

    @torch.no_grad()
    def predict(self, batch, threshold=None, signal_idx=None):
        assert (threshold is None) == (signal_idx is None), "Either both or none of threshold and signal_idx must be given"

        logits = self(batch)

        if threshold is None:
            return torch.argmax(logits, dim=1)

        probs = logits.softmax(dim=1)
        passed = probs[:, signal_idx] >= threshold
        probs_ = probs.clone()
        probs_[:, signal_idx] = -1
        best_bg = probs_.argmax(dim=1)
        return torch.where(passed, signal_idx, best_bg)


class TransformerNN(nn.Module):
    def __init__(self,
                 n_features_continuous: int,
                 categorical_sizes: list[int],
                 n_classes: int,
                 n_embed=32,
                 n_inner=None,
                 n_blocks=1,
                 n_heads=1,
                 activation=None,
                 dropout=0.3):
        super().__init__()

        if n_inner is None:
            n_inner = n_embed * 4

        self.activation = nn.GELU(approximate="tanh") if activation is None else activation
        self.embed = MyEmbed(n_features_continuous, categorical_sizes, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed, activation=self.activation, n_inner=n_inner, n_heads=n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ]) if n_blocks > 0 else nn.Identity()
        self.ln = nn.LayerNorm(n_embed)
        self.logits = nn.Linear((n_features_continuous + len(categorical_sizes)) * n_embed, n_classes)

    def forward(self, x_continuous, x_categorical):
        B = x_continuous.shape[0]

        x = self.embed(x_continuous, x_categorical)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.logits(x.view(B, -1))

        return x


class Block(nn.Module):
    def __init__(self, n_embed: int, n_inner=None, n_heads=1, activation=None, dropout=0.0):
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
            nn.Dropout(dropout)
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

        assert torch.__version__ >= "2.0.0", "MaskedAttentiveEmbed requires PyTorch 2.0.0 or higher"
        assert n_embed % n_heads == 0, f"Embedding size ({n_embed}) must be divisible by the number of heads ({n_heads})"

        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        self.dropout = dropout

        self.linear = nn.Linear(n_embed, 3 * n_embed, bias=bias)  # 3 is to have a single operation for q, k, v

        self.residual_dropout = nn.Dropout(dropout)
        self.residual_project = nn.Linear(n_embed, n_embed, bias=bias)

    def forward(self, x):
        B, T, E = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.linear(x).split(E, dim=2)

        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, E)  # re-assemble all head outputs side by side

        y = self.residual_dropout(self.residual_project(y))

        return y


class MyEmbed(nn.Module):
    def __init__(self, n_features_continuous: int, categorical_sizes: list[int], n_embed: int):
        super().__init__()

        # If we encounter NaN feature, we replace them with a learnable parameter
        self.w_nan = nn.Parameter(torch.randn(n_features_continuous))

        # Separate embedding for each categorical feature
        self.register_buffer("offsets", torch.tensor([0] + categorical_sizes[:-1]).cumsum(dim=0))
        self.w_categorical = nn.Parameter(torch.randn(sum(categorical_sizes), n_embed))
        self.b_categorical = nn.Parameter(torch.randn(len(categorical_sizes), n_embed))

        # Scale and shift for the continuous features
        self.w_continuous = nn.Parameter(torch.randn(n_features_continuous, n_embed))
        self.b_continuous = nn.Parameter(torch.randn(n_features_continuous, n_embed))

    def forward(self, x_continuous, x_categorical):
        # Embed continuous features
        # Replace NaN with a learnable parameter
        x_continuous = torch.where(torch.isnan(x_continuous), self.w_nan, x_continuous)

        # Element-wise scale and shift
        B, F = x_continuous.shape
        x_continuous = x_continuous.view(B, F, 1) * self.w_continuous + self.b_continuous

        # Embed categorical features
        x_categorical = self.w_categorical[x_categorical + self.offsets] + self.b_categorical

        # Concatenate all features
        return torch.cat([x_continuous, x_categorical], dim=1)
