import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from src.nn import Model
from src.data import Data

from .confusion_matrix import confusion_matrix


def evaluate_confusion_matrices(model: Model, val: Data, threshold=0.0, batch_size=1024, device="cpu", wandb_run=None):
    plt.rcParams.update({'font.size': 22})

    model.eval()
    model.to(device)

    # Get the probabilities
    signal_idx = val.y_names.index("ttH")
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

    # Now plot CM for just the signal class
    cm = confusion_matrix(y_pred, val.y, val.w, signal=val.y_names.index("ttH"))

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(cm.cpu(), annot=True, cmap="Blues", cbar=True,  ax=ax2, xticklabels=["Signal", "Background"], yticklabels=["Signal", "Background"], fmt=".2f")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    if wandb_run is not None:
        wandb.log({f"confusion_matrix/threshold={threshold:.3f}/all": wandb.Image(fig1)})
        wandb.log({f"confusion_matrix/threshold={threshold:.3f}/signal": wandb.Image(fig2)})