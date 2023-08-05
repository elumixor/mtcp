import os
import torch
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import trange

from ml.data import Data
from ml.nn import Model


def evaluate_rocs(model: Model, val: Data, batch_size: int, device="cpu", wandb_run=None, files_dir=None):
    plt.rcParams.update({'font.size': 22})

    # Get probs on the validation set
    with torch.no_grad():
        probs = torch.cat([model(batch.to(device)) for batch in val.batches(batch_size=batch_size, shuffle=False)]).softmax(dim=1).cpu()

    for i in trange(val.n_classes, desc="Evaluating ROC curves"):
        fpr, tpr, _ = roc_curve(val.y, probs[:, i], pos_label=i)
        auc = np.trapz(tpr, fpr)

        fprw, tprw, _ = roc_curve(val.y, probs[:, i], pos_label=i, sample_weight=val.w.cpu())
        aucw = np.trapz(tprw, fprw)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(fpr, tpr, label=f"{val.y_names[i]} {auc:.4f}")
        ax.plot(fprw, tprw, label=f"{val.y_names[i]} (weighted) {aucw:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", c="k", label="Chance")
        ax.set_ylabel("True positive rate")
        ax.set_xlabel("False positive rate")
        ax.set_title(f"ROC curve ({val.y_names[i]})")
        ax.legend()

        if wandb_run is not None:
            wandb.log({f"roc/{val.y_names[i]}": wandb.Image(fig)})

        if files_dir is not None:
            roc_dir = f"{files_dir}/roc"
            if not os.path.exists(roc_dir):
                os.makedirs(roc_dir)

            fig.savefig(f"{roc_dir}/{val.y_names[i]}.pdf")

        plt.close(fig)