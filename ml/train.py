import autorootcwd  # Do not delete - adds the root of the project to the path

import os
import torch
import wandb
import argparse
import tempfile
import signal
from tqdm import tqdm

from ml.data import load_data
from ml.nn import Transformer, ResNet
from ml.evaluation import evaluate, evaluate_rocs, evaluate_confusion_matrices, evaluate_significance, evaluate_feature_importance
from ml.training import find_lr, train, load_checkpoint
from ml.utils import get_config


# Parse the arguments
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('config', type=str, default='config.yaml', help='Path to the config file')
parser.add_argument('--tags', type=str, nargs='+', default=[], help='Tags to add to the run')
parser.add_argument('--repeat', type=int, default=None, help='Number of times to repeat the training')
args = parser.parse_args()

# Read the config
config = get_config(args.config)

# Reproducibility
torch.manual_seed(config.seed)

# Get data
data, selected = load_data(config.data_path)
data_cut = data[selected]
data_uncut = data[~selected]

classes = config.classes if "classes" in config else data.y_names
features = config.features if "features" in config else data.x_names

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

    # Take the trn_split of the selected samples
    # 20% of that is the validation set
    trn_cut, val, tst = data_cut.split(config.trn_split)
    trn_uncut = trn_cut + data_uncut

    # Here we determine the training set we're using
    assert config.cuts in ["apply", "discard"]
    trn = trn_uncut if config.cuts == "discard" else trn_cut

    # Also apply the fraction cut if needed
    if config.fraction < 1:
        n_samples = int(config.fraction * trn.n_samples)
        # Shuffle the data
        indices = torch.randperm(trn.n_samples)[:n_samples]
        trn = trn[indices]

    print()
    print(f"Training set: {trn.n_samples} samples")
    print(f"Validation set: {val.n_samples} samples")
    print()

    # Select classes
    trn = trn.select_classes(classes)
    val = val.select_classes(classes)
    tst = tst.select_classes(classes)

    using_all_features = len(features) == (len(data.x_names_categorical) + len(data.x_names_continuous))
    print(f"Using {len(features)} features{' (all)' if using_all_features else ''}")
    print()

    # Select features
    trn = trn.select_features(features)
    val = val.select_features(features)
    tst = tst.select_features(features)

    # Merge all the background classes (everything excepr ttH) into one
    if config.use_binary:
        background_classes = [c for c in trn.y_names if c != "ttH"]
        trn = trn.merge_classes(names=background_classes, new_class_name="background")
        val = val.merge_classes(names=background_classes, new_class_name="background")
        tst = tst.merge_classes(names=background_classes, new_class_name="background")


    model = get_model()

    if config.use_wandb:
        wandb_config = dict(
            trn_size=trn.n_samples,
            val_size=val.n_samples,
            n_classes=trn.n_classes,
            n_parameters=model.n_params,
            **config)

        del wandb_config["run_name"]

        tags = config.tags + args.tags
        wandb_run = wandb.init(project=config.project_name, name=config.run_name, config=wandb_config, tags=tags)
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

    lr, min_loss = find_lr(model,
                        trn,
                        Optim,
                        batch_size=config.batch_size,
                        lr_divisions=100,
                        device=config.device,
                        run=wandb_run,
                        half=config.dtype)

    print(f"Found lr={lr:.8f} with min_loss={min_loss:.8f}")
    print()

    model = get_model()

    optim = Optim(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=50000, factor=0.1, threshold=0.01, verbose=True, min_lr=1e-7)

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

        if "roc" in config["evaluations"]:
            evaluate_rocs(model, val, batch_size=config.batch_size, device=config.device, wandb_run=wandb_run)

        thresholds = [0.0]

        if "significance" in config["evaluations"]:
            print("Evaluating significance")
            threshold, threshold_simple = evaluate_significance(model, val, F,
                                                                batch_size=config.batch_size,
                                                                device=config.device,
                                                                wandb_run=wandb_run)
            thresholds = [0.0, threshold, threshold_simple]

        if "confusion_matrix" in config["evaluations"]:
            for threshold in tqdm(thresholds, desc="Evaluating confusion matrices"):
                evaluate_confusion_matrices(model, val,
                                            threshold=threshold,
                                            batch_size=config.batch_size,
                                            device=config.device,
                                            wandb_run=wandb_run)

        if "feature_importance" in config["evaluations"]:
            evaluate_feature_importance(model, val, device=config.device, wandb_run=wandb_run)

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
