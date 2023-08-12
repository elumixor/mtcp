import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from ml.data import Data
from pipeliner.utils import DotDict


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.class_weights = None
        self.use_binary = False
        self.signal_class_idx = None

    @abstractmethod
    def get_logits(self, x_continuous, x_categorical):
        pass

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loss(self, logits, y):
        if not self.use_binary:
            return F.cross_entropy(logits, y, weight=self.class_weights)

        if self.signal_class_idx is None:
            raise ValueError("signal_class_idx must be set for binary classification")

        y = (y == self.signal_class_idx).float()

        return F.binary_cross_entropy_with_logits(logits[:, self.signal_class_idx], y)

    @torch.no_grad()
    def predict(self, batch, threshold=None, signal_idx=None, return_probs=False):
        assert (threshold is None) == (
            signal_idx is None
        ), "Either both or none of threshold and signal_idx must be given"

        logits = self(batch)

        if threshold is None:
            return torch.argmax(logits, dim=1)

        probs = logits.softmax(dim=1)
        passed = probs[:, signal_idx] >= threshold
        probs_ = probs.clone()
        probs_[:, signal_idx] = -1
        best_bg = probs_.argmax(dim=1)

        labels = torch.where(passed, signal_idx, best_bg)

        if return_probs:
            return labels, probs

        return labels

    def forward(self, batch: Data, return_loss=False, return_all=False):
        logits = self.get_logits(batch.x_continuous, batch.x_categorical)

        loss = None
        if return_loss or return_all:
            loss = self.loss(logits, batch.y)

        if return_all:
            return logits, loss

        elif return_loss:
            return loss

        return logits

    @classmethod
    def from_saved(cls, path: str, device="cpu", compile=False, return_stats=False):
        saved_data = torch.load(path, map_location=device)

        # For old models
        if "n_features_continuous" not in saved_data:
            saved_data["n_features_continuous"] = len(saved_data["x_names_continuous"])

        model = cls(**saved_data)

        if compile:
            weights = saved_data["weights"]
            model = torch.compile(model)
        else:
            # Rename the weights keys: remove the "_orig_mod" prefix. This is for loading the compiled models
            weights = {k.replace("_orig_mod.", ""): v for k, v in saved_data["weights"].items()}

        model.load_state_dict(weights)
        model.eval()
        model.to(device)

        if not return_stats:
            return model

        if "stats_best" not in saved_data:
            epoch = input("Enter the epoch of the best model: ")
            epoch = int(epoch)
            saved_data["stats_best"] = [DotDict(epoch=epoch)]
            saved_data["stats_last"] = [DotDict(epoch=epoch)]
        return model, saved_data["stats_best"], saved_data["stats_last"]
