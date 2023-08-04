"""
This file is providing all needed to load the model.

Needs to define the "Model" class that fulfills the following requirements:
- "__init__(self)" constructor with no arguments
- "forward(self)" method that takes an Awkward Array as input and returns a numpy array
- "self.features" property/field that returns a list of features used by the model
"""
from __future__ import annotations

import torch
import awkward as ak
import re

from ml.nn import Transformer, ResNet
from ml.download_model import download_model


class Batch:
    def __init__(self, x_continuous, x_categorical):
        self.x_continuous = x_continuous
        self.x_categorical = x_categorical


class Model:
    def __init__(self, config):
        model_name = config.model

        path = download_model(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cls = Transformer if "transformer" in model_name else ResNet
        self.nn = cls.from_saved(path, device=self.device)

        saved_data = torch.load(path)

        self.threshold = saved_data["threshold"]

        self.x_names_continuous = saved_data["x_names_continuous"]
        self.x_names_categorical = saved_data["x_names_categorical"]

        self.n_features_continuous = len(self.x_names_continuous)
        self.categorical_sizes = saved_data["categorical_sizes"]

        self.map_categorical = saved_data["map_categorical"]

        self.y_names = saved_data["y_names"]
        self.signal_idx = self.y_names.index("ttH")

        self.mean = saved_data["mean"]
        self.std = saved_data["std"]

        # List of features used by the model
        self.features = self.x_names_continuous + self.x_names_categorical

        self.n_jets = 0
        for i, feature in enumerate(self.features):
            # If feature is like jets_XXX_Y then turn it into jets_XXX
            if re.match(r"jets_[a-zA-Z0-9]+_[0-9]+", feature):
                self.features[i] = feature[: feature.rfind("_")]
                jet_index = int(feature[feature.rfind("_") + 1:])
                self.n_jets = max(self.n_jets, jet_index + 1)
        self.features = list(set(self.features))

        # We need to create an alternative map
        mapped = self.map_categorical
        alternative_map = dict()
        for feature, mapping in mapped.items():
            alternative_map[feature] = dict()
            for key, value in mapping.items():
                value = int(value[len(feature) + 1:])
                alternative_map[feature][value] = key

        self.alternative_map = alternative_map

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
