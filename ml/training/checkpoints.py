import os
import torch
import torch.nn as nn
from typing import NamedTuple

from ml.evaluation import evaluate


Checkpoint = NamedTuple("Checkpoint", [
    ("model", str),
    ("optim", str | None),
    ("scheduler", str | None),
    ("stats", str | None),
])


def save_checkpoint(name: str,
                    model: nn.Module | dict,
                    optim: dict | torch.optim.Optimizer | None = None,
                    scheduler: dict | torch.optim.lr_scheduler.LRScheduler | None = None,
                    stats: list[evaluate] | None = None):
    # Make directories
    if not os.path.exists(name):
        os.makedirs(name)

    # Make names
    model_name = os.path.join(name, "model.pt")
    stats_name = os.path.join(name, "stats.pt")
    optim_name = os.path.join(name, "optim.pt")
    scheduler_name = os.path.join(name, "scheduler.pt")

    files = [stats_name, model_name, optim_name, scheduler_name]

    # Remove previous files
    for name in files:
        if os.path.exists(name):
            os.remove(name)

    # Save everything
    torch.save(model if isinstance(model, dict) else model.state_dict(), model_name)
    if optim:
        torch.save(optim if isinstance(optim, dict) else optim.state_dict(), optim_name)
    if scheduler:
        torch.save(scheduler if isinstance(scheduler, dict) else scheduler.state_dict(), scheduler_name)
    if stats:
        torch.save(stats, stats_name)

    result = Checkpoint(model=model_name,
                        optim=optim_name if optim else None,
                        scheduler=scheduler_name if scheduler else None,
                        stats=stats_name if stats else None)

    return result


def load_checkpoint(name: str,
                    model: nn.Module,
                    optim: torch.optim.Optimizer | None = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None):
    # Make names
    model_name = os.path.join(name, "model.pt")
    stats_name = os.path.join(name, "stats.pt")
    optim_name = os.path.join(name, "optim.pt")
    scheduler_name = os.path.join(name, "scheduler.pt")

    # Load the model
    model.load_state_dict(torch.load(model_name))

    # Load the optim
    if optim is not None:
        optim.load_state_dict(torch.load(optim_name))

    # Load the scheduler
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(scheduler_name))

    # Load the stats
    stats = torch.load(stats_name) if os.path.exists(stats_name) else None

    return stats
