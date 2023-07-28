import signal
import torch
import torch.nn as nn
import itertools
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import wandb

from ml.data import Data
from ml.nn import Model

log = logging.getLogger(__name__)


def find_lr(model: Model,
            data: Data,
            Optim,
            lre_min=-7,
            lre_max=1,
            batch_size=64,
            lr_divisions=200,
            device="cpu",
            half=None,
            run=None,
            ax=None):
    lres = torch.linspace(lre_min, lre_max, lr_divisions)
    lrs = 10 ** lres
    losses = torch.zeros_like(lrs)

    model.to(device)

    # Mixed precision and its gradient scaler
    assert half is None or half == torch.float16 or half == torch.bfloat16
    use_half = half is not None
    scaler = torch.cuda.amp.GradScaler(enabled=use_half)

    # region Register the handler for SIGINT
    interrupted = False

    def handler(_, __):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, handler)
    # endregion

    init_loss = None
    with logging_redirect_tqdm():
        batches = itertools.cycle(data.batches(batch_size))
        for i, (lr, batch) in enumerate(zip(tqdm(lrs.tolist(), "Searching for the optimal LR"), batches)):
            if interrupted:
                print("Interrupted")
                break

            optim = Optim(model.parameters(), lr)

            # Training
            with torch.autocast(device_type=device, dtype=torch.float16 if use_half or device == "cuda" else torch.bfloat16, enabled=use_half):
                loss = model(batch.to(device), return_loss=True)

            unscaled = loss.item()
            loss = scaler.scale(loss)
            loss.backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=use_half)

            loss = unscaled

            if init_loss is None:
                init_loss = loss

            if loss > 4 * init_loss:
                log.warn("Stopping early because the loss has exploded")
                break

            losses[i] = loss

    losses = losses[:i]
    lres = lres[:i]
    lrs = lrs[:i]

    # Smooth the losses - take the average of the previous 5 and next 5 losses around the current one
    smoothed_losses = torch.zeros_like(losses)

    for i, loss in enumerate(losses):
        selected = losses[max(0, i - 5):min(len(losses), i + 5)]
        smoothed = selected.mean()
        smoothed_losses[i] = smoothed

    # Find the minimum of the smoothed losses
    min_loss = smoothed_losses.min().item()
    optimal_lr = lrs[smoothed_losses.argmin()].item()

    if ax is not None:
        import numpy as np

        ax.plot(lres, losses)
        ax.plot(lres, smoothed_losses, c="tab:orange")

        # Plot the point where the loss is minimal
        ax.scatter(np.log10(optimal_lr), min_loss, c="purple", zorder=2)

    if run is not None:
        data = list(zip(lres, losses, smoothed_losses))
        table = wandb.Table(data=data, columns=["lr", "loss", "smoothed_loss"])

        run.log({
            "lr_search": wandb.plot.line(table, "lre", "loss", "LR scan"),
            "lr_search_smooth": wandb.plot.line(table, "lre", "smoothed_loss", title="LR scan (smoothed)"),
        })

    return optimal_lr, min_loss
