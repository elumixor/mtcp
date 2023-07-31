import torch
import torch.nn as nn


class MyEmbed(nn.Module):
    def __init__(self, n_features_continuous: int, categorical_sizes: list[int], n_embed: int, embed_continuous=True, use_nan_w=True):
        super().__init__()

        self.embed_continuous = embed_continuous

        # If we encounter NaN feature, we replace them with a learnable parameter
        self.w_nan = nn.Parameter(torch.randn(n_features_continuous)) if use_nan_w else 0

        # Separate embedding for each categorical feature
        self.offsets = nn.Parameter(torch.tensor([0] + categorical_sizes[:-1]).cumsum(dim=0), requires_grad=False)
        self.w_categorical = nn.Parameter(torch.randn(sum(categorical_sizes), n_embed))
        self.b_categorical = nn.Parameter(torch.randn(len(categorical_sizes), n_embed))

        if embed_continuous:
            # Scale and shift for the continuous features
            self.w_continuous = nn.Parameter(torch.randn(n_features_continuous, n_embed))
            self.b_continuous = nn.Parameter(torch.randn(n_features_continuous, n_embed))

    def forward(self, x_continuous, x_categorical):
        B, F = x_continuous.shape

        # Embed continuous features
        # Replace NaN with a learnable parameter
        x_continuous = torch.where(torch.isnan(x_continuous), self.w_nan, x_continuous)

        # Embed categorical features
        x_categorical = self.w_categorical[x_categorical + self.offsets] + self.b_categorical

        if self.embed_continuous:
            # Element-wise scale and shift
            x_continuous = x_continuous.view(B, F, 1) * self.w_continuous + self.b_continuous

            # Concatenate all features
            return torch.cat([x_continuous, x_categorical], dim=1)

        # Otherwise jsut stack the categoricals with the continuous along the feature dimension
        x_categorical = x_categorical.view(B, -1)
        return torch.cat([x_continuous, x_categorical], dim=1)

