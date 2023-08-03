import autorootcwd  # Do not delete - adds the root of the project to the path

import os
import torch
import wandb
import argparse
import tempfile
import signal

from ml.data import load_from_config
from ml.nn import Transformer, ResNet
from ml.evaluation import evaluate
from ml.training import find_lr, train, load_checkpoint
from ml.utils import get_config
from ml.evaluate import evaluate as evaluate_final


# Parse the arguments
parser = argparse.ArgumentParser(description="Train the model.")
parser.add_argument("config", type=str, default="config.yaml", help="Path to the config file")
parser.add_argument("--tags", type=str, nargs="+", default=[], help="Tags to add to the run")
parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the training")
args = parser.parse_args()

# Read the config
config = get_config(args.config)

# Reproducibility
torch.manual_seed(config.seed)

repeat = args.repeat if args.repeat is not None else config.repeat


# Model definition
def get_model():
    if config.model == "transformer":
        model = Transformer(
            trn.n_features_continuous,
            trn.categorical_sizes,
            trn.n_classes,
            n_embed=config.n_embed,
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            dropout=config.dropout,
            class_weights=trn.class_weights if config.use_weights else None,
        )

    elif config.model == "resnet":
        model = ResNet(
            trn.n_features_continuous,
            trn.categorical_sizes,
            trn.n_classes,
            n_embed=config.n_embed,
            n_blocks=config.n_blocks,
            dropout=config.dropout,
            class_weights=trn.class_weights if config.use_weights else None,
            use_embedding=config.use_embedding,
            use_nan_w=config.use_nan_w,
            n_embed_categorical=config.n_embed_categorical,
        )

    else:
        raise ValueError(f"Unknown model type {config.model}")

    model = model.to(config.device)

    if config.compile:
        model = torch.compile(model)

    return model


# Optimizer definition
def Optim(params, lr):
    return torch.optim.AdamW(params, lr=lr)


# Register the handler for SIGINT
interrupted = False


def handler(_, __):
    global interrupted
    interrupted = True


signal.signal(signal.SIGINT, handler)

# Begin trials
for trial in range(repeat):
    if interrupted:
        break

    # Get data
    trn, val, tst = load_from_config(config)

    model = get_model()

    if config.use_wandb:
        wandb_config = dict(
            trn_size=trn.n_samples,
            val_size=val.n_samples,
            n_classes=trn.n_classes,
            n_parameters=model.n_params,
            **config,
        )

        del wandb_config["run_name"]

        tags = config.tags + args.tags
        wandb_run = wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=wandb_config,
            tags=tags,
        )
        print()

        # Define min/max metrics for W&B
        wandb.define_metric("epoch")

        wandb.define_metric("trn/loss", step_metric="epoch", summary="min")
        wandb.define_metric("val/loss", step_metric="epoch", summary="min")

        wandb.define_metric("val/acc/bin", step_metric="epoch", summary="max")
        wandb.define_metric("val/acc/multi", step_metric="epoch", summary="max")

        wandb.define_metric("val/f1", step_metric="epoch", summary="max")

        wandb.define_metric("val/auc_w/ttH", step_metric="epoch", summary="max")
        wandb.define_metric("val/auc_w/mean", step_metric="epoch", summary="max")
        wandb.define_metric("sig/significance", step_metric="epoch", summary="max")
    else:
        wandb_run = None

    lr, min_loss = find_lr(
        model,
        trn,
        Optim,
        batch_size=config.batch_size,
        lr_divisions=100,
        device=config.device,
        run=wandb_run,
        half=config.dtype,
    )

    print(f"Found lr={lr:.8f} with min_loss={min_loss:.8f}")
    print()

    model = get_model()

    optim = Optim(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=50000, factor=0.1, threshold=0.01, verbose=True, min_lr=1e-7
    )

    # Train. Obtain the checkpoints of the best and the last models
    model.train()
    model.to(config.device)

    F = 1 / (1 - config.trn_split)

    evaluator = evaluate.using(
        model,
        trn,
        val,
        "ttH",
        device=config.device,
        use_tqdm=True,
        F=F,
        batch_size=config.batch_size,
        half=config.dtype,
    )

    train(
        model,
        optim,
        trn,
        evaluator,
        epochs=config.epochs,
        validate_freq=1,
        restart=True,
        use_tqdm=True,
        device=config.device,
        checkpoints_dir=config.checkpoints_dir,
        batch_size=config.batch_size,
        scheduler=scheduler,
        half=config.dtype,
        run=wandb_run,
    )

    # Load the best model
    best_path = os.path.join(config.checkpoints_dir, f"{model.name}-best")
    load_checkpoint(best_path, model)

    # Run evaluations
    if "evaluations" in config:
        print(f"Best model loaded. Evaluations: {evaluator(-1)}")
        print(f'Running the following evaluations: {config["evaluations"]}')

        thresholds = evaluate_final(
            model=model,
            val=val,
            batch_size=config.batch_size,
            device=config.device,
            wandb_run=wandb_run,
            F=F,
            roc="roc" in config.evaluations,
            significance="significance" in config.evaluations,
            confusion_matrix="confusion_matrix" in config.evaluations,
            feature_importance="feature_importance" in config.evaluations,
        )

    # Save everything to create the NN:
    print("Saving the model")

    # - state dict (weights)
    # - feature names used
    # - class names used
    # - mean and std of the features
    saved_data = {
        **model.hyperparameters,
        "weights": model.state_dict(),
        "x_names_continuous": trn.x_names_continuous,
        "x_names_categorical": trn.x_names_categorical,
        "n_features_continuous": trn.n_features_continuous,
        "categorical_sizes": trn.categorical_sizes,
        "map_categorical": trn.metadata["map_categorical"],
        "threshold": (thresholds[1] if len(thresholds) > 1 else 0.0) if "evaluations" in config else None,
        "y_names": trn.y_names,
        "mean": trn.metadata["mean"],
        "std": trn.metadata["std"],
    }

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_path = tmp_file.name

        # Save to the temporary directory
        torch.save(saved_data, tmp_file_path)

        # Add this artifact to W&B
        artifact = wandb.Artifact("saved_data", type="model")
        artifact.add_file(tmp_file_path)
        wandb_run.log_artifact(artifact)

    # Finish the run
    wandb_run.finish()

    # Delete the temporary file
    os.remove(tmp_file_path)
