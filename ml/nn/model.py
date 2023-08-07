import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from ml.data import Data


class Model(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: Data, return_loss=False, return_all=False):
        pass

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

    @classmethod
    def from_saved(cls, path: str, device="cpu"):
        saved_data = torch.load(path, map_location=device)

        # For old models
        if "n_features_continuous" not in saved_data:
            saved_data["n_features_continuous"] = len(saved_data["x_names_continuous"])

        model = cls(**saved_data)
        # Rename the weights keys: remove the "_orig_mod" prefix. This is for loading the compiled models
        weights = {k.replace("_orig_mod.", ""): v for k, v in saved_data["weights"].items()}

        model.load_state_dict(weights)
        model.eval()
        model.to(device)

        return model
