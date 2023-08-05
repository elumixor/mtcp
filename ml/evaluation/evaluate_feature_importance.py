import os
import wandb
import torch
import matplotlib.pyplot as plt

from ml.nn import Model
from ml.data import Data

def evaluate_feature_importance(model: Model, val: Data, device="cpu", wandb_run=None, files_dir=None, num_examples=10):
    from ml.evaluation import feature_importance

    plt.rcParams.update({'font.size': 10})

    torch.manual_seed(0)

    ig, fig1 = feature_importance(model, val, val.y_names.index("ttH"), val.x_names, device=device, num_examples=num_examples, plot="horizontal", return_fig=True)
    fig1.tight_layout()

    # Take top 20:
    plt.rcParams.update({'font.size': 20})

    n_features = 20
    # Now we can plot the integrated gradients
    # For now let's drop the object feature (our simple model doesn't use it anyway)
    fig2 = plt.figure(figsize=(20, 10))
    ig_sorted, indices = torch.sort(ig.abs(), descending=False)
    ig_sorted = ig[indices][-n_features:] # Use this if you also want to see positive/negative
    indices = indices[-n_features:]

    plt.bar(torch.arange(len(ig_sorted)), ig_sorted.abs().cpu().tolist()[::-1])
    plt.xticks(torch.arange(len(ig_sorted)), [val.x_names[i] for i in indices.tolist()[::-1]], rotation=90);
    fig2.tight_layout()

    # Now also sort in ascending order
    ig_sorted, ii = torch.sort(ig_sorted, descending=True)
    indices = indices[ii]

    # ig_sorted = ig_sorted[-n_features:]
    fig3 = plt.figure(figsize=(20, 10))
    plt.bar(torch.arange(len(ig_sorted)), ig_sorted.cpu())
    plt.xticks(torch.arange(len(ig_sorted)), [val.x_names[i] for i in indices], rotation=90);
    fig3.tight_layout()

    if wandb_run is not None:
        wandb.log({"feature_importance/all": wandb.Image(fig1)})
        wandb.log({"feature_importance/top20-abs": wandb.Image(fig2)})
        wandb.log({"feature_importance/top20": wandb.Image(fig3)})

    if files_dir is not None:
        feature_dir = f"{files_dir}/feature_importance"
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        fig1.savefig(f"{feature_dir}/all.pdf")
        fig2.savefig(f"{feature_dir}/top20-abs.pdf")
        fig3.savefig(f"{feature_dir}/top20.pdf")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)