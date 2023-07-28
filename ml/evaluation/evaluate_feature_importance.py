import wandb
import torch
import matplotlib.pyplot as plt

from ml.nn import Model
from ml.data import Data

def evaluate_feature_importance(model: Model, val: Data, device="cpu", wandb_run=None):
    from ml.evaluation import feature_importance

    plt.rcParams.update({'font.size': 10})

    torch.manual_seed(0)

    ig, fig = feature_importance(model, val, val.y_names.index("ttH"), val.x_names, device=device, num_examples=10, plot="horizontal", return_fig=True)
    fig.tight_layout()

    if wandb_run is not None:
        wandb.log({"feature_importance/all": wandb.Image(fig)})

    # Take top 20:
    plt.rcParams.update({'font.size': 20})

    n_features = 20
    # Now we can plot the integrated gradients
    # For now let's drop the object feature (our simple model doesn't use it anyway)
    fig = plt.figure(figsize=(20, 10))
    ig_sorted, indices = torch.sort(ig.abs(), descending=False)
    ig_sorted = ig[indices][-n_features:] # Use this if you also want to see positive/negative
    indices = indices[-n_features:]

    plt.bar(torch.arange(len(ig_sorted)), ig_sorted.abs().cpu().tolist()[::-1])
    plt.xticks(torch.arange(len(ig_sorted)), [val.x_names[i] for i in indices.tolist()[::-1]], rotation=90);
    fig.tight_layout()

    if wandb_run is not None:
        wandb.log({"feature_importance/top20-abs": wandb.Image(fig)})

    # Now also sort in ascending order
    ig_sorted, ii = torch.sort(ig_sorted, descending=True)
    indices = indices[ii]

    # ig_sorted = ig_sorted[-n_features:]
    fig = plt.figure(figsize=(20, 10))
    plt.bar(torch.arange(len(ig_sorted)), ig_sorted.cpu())
    plt.xticks(torch.arange(len(ig_sorted)), [val.x_names[i] for i in indices], rotation=90);
    fig.tight_layout()

    if wandb_run is not None:
        wandb.log({"feature_importance/top20": wandb.Image(fig)})
