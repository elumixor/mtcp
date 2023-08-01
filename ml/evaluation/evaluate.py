from __future__ import annotations
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from ml.data import Data
from ml.nn import Model

from .significance import significance
from .confusion_matrix import confusion_matrix, make_binary_cm
from .find_significance_threshold import find_significance_threshold


class evaluate:
    def __init__(self,
                 model: Model,
                 trn: Data,
                 val: Data,
                 signal_idx: int,
                 epoch: int,
                 F: float,
                 batch_size: int,
                 max_significance: float,
                 use_tqdm=True,
                 priority=None,
                 n_significance_thresholds=100,
                 half=None,
                 device="cpu"):
        if priority is None:
            priority = ["val/auc_w/ttH", "sig/significance", "val/f1", "val/acc", "val/loss", "val/loss"]

        self.metrics = dict()
        self.metrics["epoch"] = epoch

        self.priority = priority
        self.metrics["sig/max_significance"] = max_significance

        model.eval()
        model.to(device)

        assert half is None or half == torch.float16 or half == torch.bfloat16
        use_half = half is not None

        with torch.no_grad():
            # Compute the loss on the training set
            batches = tqdm(trn.batches(batch_size, shuffle=False), desc=" - evaluation: trn", disable=not use_tqdm)

            with torch.autocast(device_type=device, dtype=torch.float16 if use_half or device == "cuda" else torch.bfloat16, enabled=use_half):
                self.metrics["trn/loss"] = torch.stack([model(batch.to(device), return_loss=True) for batch in batches]).mean().item()

            # Evaluate the fixed model on the validation set
            batches = tqdm(val.batches(batch_size, shuffle=False), desc=" - evaluation: val", disable=not use_tqdm)
            with torch.autocast(device_type=device, dtype=torch.float16 if use_half or device == "cuda" else torch.bfloat16, enabled=use_half):
                logits, loss = zip(*[model(batch.to(device), return_all=True) for batch in batches])

                # Probabilities
                probs = torch.cat(logits, dim=0).softmax(dim=1)

            self.metrics["val/loss"] = torch.stack(loss).mean().item()

            y = val.y.to(device)
            w = val.w.to(device)

            # Now let's check all the thresholds where the significance is the highest
            self.metrics["sig/threshold"], self.metrics["sig/significance"] = find_significance_threshold(
                probs,
                y,
                w,
                signal_idx,
                F,
                n_significance_thresholds=n_significance_thresholds,
            )
            self.metrics["sig/significance_percent"] = self.metrics["sig/significance"] / max_significance

            # Now let's check all the thresholds where the significance is the highest
            self.metrics["sig/threshold_simple"], self.metrics["sig/significance_simple"] = find_significance_threshold(
                probs,
                y,
                w,
                signal_idx,
                F,
                n_significance_thresholds=n_significance_thresholds,
                include_signal=False,
            )

            # Compute the accuracy using the CM with the argmax strategy
            y_pred = probs.argmax(dim=1)
            cm = confusion_matrix(y_pred, y, w, n_classes=val.n_classes)
            cm_binary = make_binary_cm(cm, signal_idx)

            # Compute the accuracy (multiclass)
            self.metrics["val/acc/multi"] = cm.trace() / cm.sum()

            # Compute the raw (unweighted) accuracy (multiclass)
            self.metrics["val/acc/multi_raw"] = (y_pred == y).float().mean().item()

            # Compute the accuracy (binary)
            self.metrics["val/acc/bin"] = cm_binary.trace() / cm_binary.sum()

            # Compute the raw (unweighted) accuracy (binary)
            self.metrics["val/acc/bin_raw"] = ((y_pred == signal_idx) == (y == signal_idx)).float().mean().item()

            # Compute F1 score
            tp = cm_binary[0, 0]
            fp = cm_binary[1, 0]
            fn = cm_binary[0, 1]
            self.metrics["val/f1"] = 2 * tp / (2 * tp + fp + fn)

            # Compute the AUCs
            aucs = []
            aucsw = []
            probs = probs.cpu()
            for i in range(val.n_classes):
                # Skip if there are no examples of this class
                if (val.y == i).sum() == 0:
                    continue

                fpr, tpr, _ = roc_curve(val.y, probs[:, i], pos_label=i)
                auc = np.trapz(tpr, fpr)

                fprw, tprw, _ = roc_curve(val.y, probs[:, i], pos_label=i, sample_weight=val.w)
                aucw = np.trapz(tprw, fprw)

                self.metrics[f"val/auc/{val.y_names[i]}"] = auc
                self.metrics[f"val/auc_w/{val.y_names[i]}"] = aucw

                aucs.append(auc)
                aucsw.append(aucw)

            # Mean auroc
            self.metrics["val/auc/mean"] = np.mean(aucs)
            self.metrics["val/auc_w/mean"] = np.mean(aucsw)

        model.train()

    def to_dict(self):
        return self.metrics

    @property
    def epoch(self):
        return self.metrics["epoch"]

    @property
    def scheduler_metric(self):
        return self.metrics["val/auc/ttH"]

    def __gt__(self, other):
        assert isinstance(other, evaluate)

        for key in self.priority:
            # Lower loss is better
            if key.startswith("loss"):
                return self.metrics[key] < other.metrics[key]
            # Higher accuracy is better
            else:
                return self.metrics[key] > other.metrics[key]

        return False

    def __str__(self):
        s = f"Epoch {self.metrics['epoch']}: " + \
            f"val/loss={self.metrics['val/loss']:.4f}, " + \
            f"trn/loss={self.metrics['trn/loss']:.4f}, " + \
            f"val/acc={self.metrics['val/acc/multi']:.2%}, " + \
            f"val/acc/bin={self.metrics['val/acc/bin']:.2%}, " + \
            f"val/f1={self.metrics['val/f1']:.2%}, " + \
            f"AUC (mean)={self.metrics['val/auc_w/mean']:.3f}, " + \
            f"AUC (ttH)={self.metrics['val/auc_w/ttH']:.3f}, " + \
            f"significance={self.metrics['sig/significance']:.2f} " + \
            f"({self.metrics['sig/significance_percent']:.2%} of max possible " + \
            f"({self.metrics['sig/max_significance']:.2f})) @ threshold={self.metrics['sig/threshold']:.2f}"

        return s

    @staticmethod
    def using(model: nn.Module, trn: Data, val: Data, signal: int | str, batch_size: int, F: float | None = None, **kwargs):
        F = (trn.n_samples + val.n_samples) / val.n_samples if F is None else F
        signal_idx = signal if isinstance(signal, int) else val.y_names.index(signal)
        perfect_cm = confusion_matrix(val.y, val.y, val.w, signal=signal_idx)
        max_significance = significance(perfect_cm, F=F)

        return lambda epoch: evaluate(
            model,
            trn,
            val,
            signal_idx,
            epoch,
            F,
            batch_size,
            max_significance=max_significance,
            **kwargs
        )

    @classmethod
    def plot(cls, stats: list[evaluate], ax1, ax2):
        ax1.plot([stat.metrics["val/loss"] for stat in stats], label="Validation")
        ax1.plot([stat.metrics["trn/loss"] for stat in stats], label="Training")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot([stat.metrics["val/acc_bin"] for stat in stats], label="Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
