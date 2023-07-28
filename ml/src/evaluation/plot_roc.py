import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from src.data import Data
from src.nn import Model


def plot_roc(model: Model, val: Data, class_idx=None, device="cpu", batch_size=1024, use_weights=False):
    model.eval()
    model.to(device)

    # Get the probabilities
    with torch.no_grad():
        probs = torch.cat([model(batch.to(device)).softmax(dim=1) for batch in val.batches(batch_size=batch_size, shuffle=False)], dim=0).cpu()

    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot the dashed "chance" diagonal line
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")

    class_indices = [class_idx] if class_idx is not None else range(val.n_classes)

    rocs = []
    for class_idx in class_indices:
        # Skip if no samples of that class
        if (val.y == class_idx).sum() == 0:
            continue

        # Let's plot the ROC curves for all the classes
        fpr, tpr, _ = roc_curve(val.y, probs[:, class_idx], pos_label=class_idx, sample_weight=val.w if use_weights else None)
        auc = np.trapz(tpr, fpr)
        rocs.append((auc, fpr, tpr, val.y_names[class_idx]))

    # Sort the ROC curves by AUC
    rocs.sort(key=lambda x: x[0], reverse=True)  # Sort by AUC

    alpha = 1
    for auci, fpri, tpri, label in rocs:
        ax.plot(fpri, tpri, label=f"{label} (AUC={auci:.3f})", alpha=alpha)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        alpha *= 0.9

    # Add the legend
    ax.legend()
