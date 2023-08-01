"""
This file is providing all needed to load the model.

Needs to define the "Model" class that fulfills the following requirements:
- "__init__(self)" constructor with no arguments
- "forward(self)" method that takes an Awkward Array as input and returns a numpy array
- "self.features" property/field that returns a list of features used by the model
"""
from __future__ import annotations

import torch
import os
import awkward as ak
import re

from ml.nn import Transformer


class Batch:
    def __init__(self, x_continuous, x_categorical):
        self.x_continuous = x_continuous
        self.x_categorical = x_categorical


class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the weights stored at the same directory as this file
        dirname = os.path.dirname(__file__)
        saved_data = torch.load(os.path.join(dirname, "model.pt"), map_location=self.device)

        self.x_names_continuous = saved_data["x_names_continuous"]
        self.x_names_categorical = saved_data["x_names_categorical"]

        self.n_features_continuous = len(self.x_names_continuous)
        self.categorical_sizes = saved_data["categorical_sizes"]

        self.map_categorical = saved_data["map_categorical"]

        self.y_names = saved_data["y_names"]
        self.signal_idx = self.y_names.index("ttH")

        n_classes = len(self.y_names)

        self.mean = saved_data["mean"]
        self.std = saved_data["std"]

        self.n_jets = 6  # TODO: take this from config

        n_blocks = saved_data["n_blocks"]
        n_embed = saved_data["n_embed"]
        n_heads = saved_data["n_heads"]
        n_inner = saved_data["n_inner"]
        self.threshold = saved_data["threshold"]
        if self.threshold is None:
            self.threshold = 0.5

        self.nn = Transformer(
            self.n_features_continuous,
            self.categorical_sizes,
            n_classes,
            n_embed,
            n_inner,
            n_blocks,
            n_heads,
        )

        # Move to GPU if available
        self.nn.to(self.device)

        # List of features used by the model
        self.features = self.x_names_continuous + self.x_names_categorical

        for i, feature in enumerate(self.features):
            # If feature is like jets_XXX_Y then turn it into jets_XXX
            if re.match(r"jets_[a-zA-Z0-9]+_[0-9]+", feature):
                self.features[i] = feature[: feature.rfind("_")]
        self.features = list(set(self.features))

        # Rename the weights keys: remove the "_orig_mod" prefix. This is for loading the compiled models
        weights = {k.replace("_orig_mod.", ""): v for k, v in saved_data["weights"].items()}
        self.nn.load_state_dict(weights)

        # We need to create an alternative map
        mapped = self.map_categorical
        alternative_map = dict()
        for feature, mapping in mapped.items():
            alternative_map[feature] = dict()
            for key, value in mapping.items():
                value = int(value[len(feature) + 1:])
                alternative_map[feature][value] = key

        self.alternative_map = alternative_map

        # Set to eval mode
        self.nn.eval()

    @torch.no_grad()
    def __call__(self, x: ak.Array):
        """Forward pass of the NN. Note that the input is an Awkward Array"""
        # Preprocess the root arrays
        batch = self.preprocess(x)

        # Run the network to obtain the logits
        predictions = self.nn.predict(batch, threshold=self.threshold, signal_idx=self.signal_idx)

        # Return the predictions as numpy arrays
        return predictions.cpu().numpy()

    def preprocess(self, x: ak.Array):
        """Preprocess the root arrays"""

        x_continuous = torch.zeros((len(x), self.n_features_continuous), dtype=torch.float32, device=self.device)
        x_categorical = torch.zeros((len(x), len(self.categorical_sizes)), dtype=torch.long, device=self.device)

        # Process continuous features
        for i, feature in enumerate(self.x_names_continuous):
            if not re.match(r"jets_[a-zA-Z0-9]+_[0-9]+", feature):
                feature = torch.from_numpy(x[feature].to_numpy()).to(self.device)
            else:
                feature_i = int(feature[feature.rfind("_") + 1:])
                feature_name = feature[: feature.rfind("_")]
                feature_data = x[feature_name]

                max_length = max(ak.num(feature_data))
                # Make sure we have exactly the same number of objects in each event
                max_length = max(max_length, self.n_jets)
                feature_data = ak.pad_none(x[feature_name], max_length).to_numpy()

                feature = torch.from_numpy(feature_data[:, feature_i]).to(self.device)

            feature = (feature - self.mean[i]) / self.std[i]
            x_continuous[:, i] = feature

        # Process categorical features
        for i, feature in enumerate(self.x_names_categorical):
            mapped = self.alternative_map[feature]
            feature = x[feature].to_numpy()

            for key, value in mapped.items():
                feature[feature == key] = value

        x_continuous = x_continuous.to(self.device)
        x_categorical = x_categorical.to(self.device)

        return Batch(x_continuous, x_categorical)
