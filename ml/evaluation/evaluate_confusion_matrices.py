import os
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from ml.nn import Model
from ml.data import Data

from .confusion_matrix import confusion_matrix


def evaluate_confusion_matrices(model: Model, val: Data, threshold=0.0, batch_size=1024, device="cpu", wandb_run=None, files_dir=None):
    plt.rcParams.update({'font.size': 22})

    model.eval()
    model.to(device)

    # Get the probabilities
    signal_idx = val.y_names.index("ttH")

    if threshold is None:
        # Use argmax
        y_pred = torch.cat([
            model.predict(batch.to(device))
            for batch in val.batches(batch_size=batch_size, shuffle=False)
        ]).cpu()
    else:
        # Use threshold
        y_pred = torch.cat([
            model.predict(batch.to(device), threshold=threshold, signal_idx=signal_idx)
            for batch in val.batches(batch_size=batch_size, shuffle=False)
        ]).cpu()

    # Plot CM for all the classes
    cm = confusion_matrix(y_pred, val.y, val.w, n_classes=val.n_classes)

    fig1, ax1 = plt.subplots(figsize=(25, 15))
    sns.heatmap(cm.cpu(), annot=True, cmap="Blues", cbar=True,  ax=ax1, xticklabels=val.y_names, yticklabels=val.y_names, fmt=".2f")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    fig1.tight_layout()

    # Now plot CM for just the signal class
    cm = confusion_matrix(y_pred, val.y, val.w, signal=val.y_names.index("ttH"))

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(cm.cpu(), annot=True, cmap="Blues", cbar=True,  ax=ax2, xticklabels=["Signal", "Background"], yticklabels=["Signal", "Background"], fmt=".2f")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    fig2.tight_layout()

    threshold_str = "argmax" if threshold is None else f"threshold={threshold:.3f}"

    if wandb_run is not None:
        wandb.log({f"confusion_matrix/{threshold_str}/all": wandb.Image(fig1)})
        wandb.log({f"confusion_matrix/{threshold_str}/signal": wandb.Image(fig2)})

    if files_dir is not None:
        cm_dir = f"{files_dir}/confusion_matrix"
        if not os.path.exists(cm_dir):
            os.makedirs(cm_dir)

        fig1.savefig(f"{cm_dir}/all_{threshold_str}.pdf")
        fig2.savefig(f"{cm_dir}/signal_{threshold_str}.pdf")

    # Close the figures
    plt.close(fig1)
    plt.close(fig2)