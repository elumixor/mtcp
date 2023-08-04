import os
import signal
import logging
import time
from typing import Callable
import torch
import wandb
from tqdm import tqdm
from copy import deepcopy

from ml.evaluation import evaluate
from ml.data import Data
from ml.nn import Model

from .checkpoints import save_checkpoint, load_checkpoint

log = logging.getLogger(__name__)


def train(
    model: Model,
    optim: torch.optim.Optimizer,
    trn: Data,
    evaluate_fn: Callable[[int], evaluate],
    epochs: int,
    validate_freq: int,
    scheduler=None,
    batch_size=64,
    checkpoints_dir="ml/checkpoints",
    run=None,
    restart=False,
    device="cpu",
    use_tqdm=True,
    half=None,
):
    # Let's estimate the initial loss
    initial_loss = model.estimated_initial_loss
    log.debug(f"Initial loss should be somewhere around {initial_loss:.5f}")
    log.debug(f"Model has {model.n_params} parameters")

    # Mixed precision and its gradient scaler
    assert half is None or half == torch.float16 or half == torch.bfloat16
    use_half = half is not None
    scaler = torch.cuda.amp.GradScaler(enabled=use_half)

    # First, let's load the checkpoint
    best_path = os.path.join(checkpoints_dir, f"{model.name}-best")
    last_path = os.path.join(checkpoints_dir, f"{model.name}-last")
    try:
        if restart:
            raise FileNotFoundError

        stats_best = load_checkpoint(best_path, model, optim, scheduler)
        stats = load_checkpoint(last_path, model, optim, scheduler)

        print(f"Last evaluation:\n{stats[-1]}\n")
    except FileNotFoundError:
        # If there's no checkpoint, create the initial checkpoint for the best and for the last evaluations
        evaluation = evaluate_fn(0)
        stats_best = [evaluation]
        stats = [evaluation]

        save_checkpoint(best_path, model, optim, scheduler, stats_best)
        save_checkpoint(last_path, model, optim, scheduler, stats)

        # Print the initial/best evaluation
        print(f"Initial evaluation:\n{evaluation}\n")

    # Get the best evaluation from the best checkpoint
    evaluation_best = stats_best[-1]
    model_best = model.state_dict()
    optim_best = optim.state_dict()
    scheduler_best = scheduler.state_dict() if scheduler is not None else None

    interrupted = False

    # Register the handler for SIGINT
    def handler(_, __):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)

    # Track the time between the epochs
    start_time = time.time()

    model.train()
    first_epoch = stats[-1].epoch + 1
    for epoch in range(first_epoch, first_epoch + epochs):
        batches = trn.batches(batch_size)
        for batch in tqdm(
            batches, desc=f"Training (epoch {epoch})", disable=not use_tqdm
        ):
            if interrupted:
                break

            # Training
            with torch.autocast(
                device_type=device,
                dtype=torch.float16 if use_half or device == "cuda" else torch.bfloat16,
                enabled=use_half,
            ):
                loss = model(batch.to(device), return_loss=True)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=use_half)

        if interrupted:
            break

        # Validation
        if epoch % validate_freq == 0 or epoch == epochs - 1:
            evaluation = evaluate_fn(epoch)
            stats.append(evaluation)

            if scheduler is not None:
                scheduler.step(evaluation.scheduler_metric)

            # Print the current evaluation
            end_time = time.time()
            seconds = end_time - start_time
            start_time = end_time
            print(f"+{seconds:5.0f}s: {evaluation}")

            # Log stats to wandb
            if run is not None:
                metrics = evaluation.to_dict()
                metrics["lr"] = optim.param_groups[0]["lr"]
                metrics["n_grad_steps"] = epoch * len(batches)
                run.log(metrics)

            if evaluation > evaluation_best:
                evaluation_best = evaluation
                stats_best = stats[:]

                # Record the best data (we need to make a deep copy, otherwise the best data will be overwritten)
                model_best = deepcopy(model.state_dict())
                optim_best = deepcopy(optim.state_dict())
                scheduler_best = (
                    deepcopy(scheduler.state_dict())
                    if scheduler is not None else None
                )

            # Add the SIGINT handler again because the evaluation uses the same handler
            def handler(_, __):
                nonlocal interrupted
                interrupted = True

            signal.signal(signal.SIGINT, handler)

    # Save the checkpoints - best and last
    checkpoint_last = save_checkpoint(last_path, model, optim, scheduler, stats)
    checkpoint_best = save_checkpoint(
        best_path, model_best, optim_best, scheduler_best, stats_best
    )

    if run is not None:
        # Log the artifacts
        for kind, checkpoint in [("last", checkpoint_last), ("best", checkpoint_best)]:
            artifact = wandb.Artifact(f"checkpoint-{kind}", type="checkpoint")
            artifact.add_file(checkpoint.model)

            if checkpoint.optim is not None:
                artifact.add_file(checkpoint.optim)

            if checkpoint.scheduler is not None:
                artifact.add_file(checkpoint.scheduler)

            run.log_artifact(artifact)

    # Return the stats
    return stats, checkpoint_best, checkpoint_last
