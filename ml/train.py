from copy import deepcopy
import os
import torch
import logging
import wandb
import yaml
from tqdm import tqdm
import argparse

import autorootcwd  # Do not delete - adds the root of the project to the path

from ml.data import load_data
from ml.nn import Transformer, ResNet
from ml.evaluation import evaluate, evaluate_rocs, evaluate_confusion_matrices, evaluate_significance, evaluate_feature_importance
from ml.training import find_lr, train, load_checkpoint

# Setup logging
log = logging.getLogger()
log.setLevel(logging.INFO)

logging.basicConfig(format="[{levelname:.1s}] {funcName:^20s} :: {message}", style="{")

parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
args = parser.parse_args()

# Hyperparameters (read from config file)
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda" and "require_cuda" in config and config["require_cuda"]:
    raise ValueError("CUDA is not available but is required")

if device == "cuda":
    # Make sure that CUDA env is provided
    assert "CUDA_VISIBLE_DEVICES" in os.environ, "CUDA_VISIBLE_DEVICES not set"
    cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]

    device = f"cuda:{cuda_devices}"

dtype = torch.float16
use_compile = torch.cuda.get_device_capability()[0] >= 7
checkpoints_dir = config["checkpoints_dir"] if "checkpoints_dir" in config else "checkpoints"

instance_configs = []
defaults = config["defaults"]
variations = config["variations"] if "variations" in config else {}

for parameter, values in variations.items():
    if isinstance(values, list):
        print(f"{parameter}: {len(values)} values")
    elif isinstance(values, dict):
        print(f"{parameter}: {len(values.keys())} values")

# Generate permutations of the hyperparameters
instance_configs = [deepcopy(defaults)]
for parameter, values in variations.items():
    if isinstance(values, list):
        instance_configs = [
            {
                **deepcopy(config),
                parameter: value,
                "name": f"{config['name']} {parameter}={'(' + str(len(value)) + ' values)' if isinstance(value, list) else value}"
            } for config in instance_configs for value in values
        ]
    elif isinstance(values, dict):
        instance_configs = [
            {
                **deepcopy(config),
                parameter: value,
                "name": f"{config['name']} {parameter}={key}"
            } for config in instance_configs for key, value in values.items()
        ]

print(f"Total number of configurations: {len(instance_configs)}")

for config in instance_configs:
    wandb_run = config["name"]
    print(wandb_run)

    seed = config["seed"]
    n_blocks = config["n_blocks"]
    n_heads = config["n_heads"]
    n_embed = config["n_embed"]
    dropout = config["dropout"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    use_weights = config["use_weights"]
    merge_background = config["merge_background"]
    data_path = config["data_path"] if "data_path" in config else "data_processing/output"
    trn_split = config["trn_split"]
    model_type = config["model_type"]
    fraction = config["fraction"] if "fraction" in config else 1

    torch.manual_seed(seed)  # For reproducibility

    # Print all the hyperparameters
    hyperparameters = {
        **config,
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "n_embed": n_embed,
        "dropout": dropout,
        "batch_size": batch_size,
        "device": device,
        "dtype": str(dtype),
        "seed": seed,
        "compiled?": use_compile,
        "use_weights": use_weights,
        "merge_background": merge_background,
    }

    # Get data
    data, selected = load_data(data_path)

    # Apply the cut on the nJets and other cuts if needed
    cuts = config["cuts"] if "cuts" in config else {}
    nJets_OR_cut = cuts["nJets_OR"] if "nJets_OR" in cuts else None
    other_cuts = cuts["other"] if "other" in cuts else "discard"

    # selected should always apply the nJets_OR >= 4 cut
    feature_idx = data.x_names_continuous.index("nJets_OR")
    mean = data.metadata["mean"][feature_idx]
    std = data.metadata["std"][feature_idx]
    additional_cut = (data.x_continuous[:, feature_idx] * std + mean) >= 4

    # Take the trn_split of the selected samples
    data_cut = data[selected & additional_cut]
    data_uncut = data[~selected]
    # 20% of that is the validation set
    trn_cut, val, tst = data_cut.split(trn_split)

    trn_uncut = trn_cut + data_uncut

    # Here we determine the training set we're using
    assert other_cuts in ["apply", "discard"]
    trn = trn_uncut if other_cuts == "discard" else trn_cut + data[selected & ~additional_cut]

    # Additionally apply the cut on nJets if needed
    if nJets_OR_cut is not None:
        threshold = (trn.x_continuous[:, feature_idx] * std + mean) >= nJets_OR_cut

        # Assert that at least one sample remains
        assert threshold.sum() > 0, "No samples remain after the nJets_OR cut"

        trn = trn[threshold]

    # Also apply the fraction cut if needed
    if fraction < 1:
        trn = trn[:int(fraction * trn.n_samples)]

    print(f"Training on {trn.n_samples} samples, validating on {val.n_samples}")

    # Select classes
    classes = config["classes"] if "classes" in config else trn.y_names
    trn = trn.select_classes(classes)
    val = val.select_classes(classes)
    tst = tst.select_classes(classes)

    # Select features
    allowed_features = config["features"] if "features" in config else trn.x_names
    trn = trn.select_features(allowed_features)
    val = val.select_features(allowed_features)
    tst = tst.select_features(allowed_features)

    # Merge all the background classes (everything excepr ttH) into one
    if merge_background:
        background_classes = [c for c in trn.y_names if c != "ttH"]
        trn = trn.merge_classes(names=background_classes, new_class_name="background")
        val = val.merge_classes(names=background_classes, new_class_name="background")
        tst = tst.merge_classes(names=background_classes, new_class_name="background")

    # Model definition

    def get_model():
        if model_type == "transformer":
            model = Transformer(
                trn.n_features_continuous,
                trn.categorical_sizes,
                trn.n_classes,
                n_embed=n_embed,
                n_blocks=n_blocks,
                n_heads=n_heads,
                dropout=dropout,
                class_weights=trn.class_weights if use_weights else None,
            )

        elif model_type == "resnet":
            model = ResNet(
                trn.n_features,
                trn.n_classes,
                embed_nan=True,
                n_embed=n_embed,
                n_blocks=n_blocks,
                dropout=dropout,
                class_weights=trn.class_weights if use_weights else None,
            )

        else:
            raise ValueError(f"Unknown model type {model_type}")

        model = model.to(device)

        if use_compile:
            model = torch.compile(model)

        return model

    # Optimizer definition

    def Optim(params, lr):
        return torch.optim.AdamW(params, lr=lr)

    model = get_model()

    if wandb_run is not None:
        config = {}
        config["trn_size"] = trn.n_samples
        config["val_size"] = val.n_samples
        config["n_classes"] = trn.n_classes

        for key, value in model.hyperparameters.items():
            config[key] = value

        if hyperparameters is not None:
            for key, value in hyperparameters.items():
                config[key] = value

        wandb_run = wandb.init(project="ttH", name=wandb_run, config=config)
    else:
        wandb_run = None

    lr, min_loss = find_lr(model,
                           trn,
                           Optim,
                           batch_size=batch_size,
                           lr_divisions=100,
                           device=device,
                           run=wandb_run,
                           half=dtype)

    print(f"Found lr={lr:.8f} with min_loss={min_loss:.8f}")

    model = get_model()

    optim = Optim(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=50000, factor=0.1, threshold=0.01, verbose=True, min_lr=1e-7)

    # Train. Obtain the checkpoints of the best and the last models
    model.train()
    model.to(device)

    F = 1 / (1 - trn_split)

    evaluator = evaluate.using(
        model,
        trn,
        val,
        "ttH",
        device=device,
        use_tqdm=True,
        F=F,
        batch_size=batch_size,
        half=dtype,
    )

    train(
        model,
        optim,
        trn,
        evaluator,
        epochs=epochs,
        validate_freq=1,
        restart=True,
        use_tqdm=True,
        device=device,
        checkpoints_dir=checkpoints_dir,
        batch_size=batch_size,
        scheduler=scheduler,
        half=dtype,
        run=wandb_run,
    )

    # Load the best model
    best_path = os.path.join(checkpoints_dir, f"{model.name}-best")
    load_checkpoint(best_path, model)

    # Run evaluations
    if "evaluations" in config:
        print(f"Best model loaded. Evaluations: {evaluator(-1)}")

        print(f'Running the following evaluations: {config["evaluations"]}')

        if "roc" in config["evaluations"]:
            evaluate_rocs(model, val, batch_size=batch_size, device=device, wandb_run=wandb_run)

        thresholds = [0.0]

        if "significance" in config["evaluations"]:
            print("Evaluating significance")
            threshold, threshold_simple = evaluate_significance(model, val, F, batch_size=batch_size, device=device, wandb_run=wandb_run)
            thresholds = [0.0, threshold, threshold_simple]

        if "confusion_matrix" in config["evaluations"]:
            for threshold in tqdm(thresholds, desc="Evaluating confusion matrices"):
                evaluate_confusion_matrices(model, val, threshold=threshold, batch_size=batch_size, device=device, wandb_run=wandb_run)

        if "feature_importance" in config["evaluations"]:
            evaluate_feature_importance(model, val, device=device, wandb_run=wandb_run)

    # Save everything to create the NN:
    print("Saving the model")
    # - state dict (weights)
    # - feature names used
    # - class names used
    # - mean and std of the features
    saved_data = {
        "weights": model.state_dict(),
        "x_names_continuous": trn.x_names_continuous,
        "x_names_categorical": trn.x_names_categorical,
        "categorical_sizes": trn.categorical_sizes,
        "map_categorical": trn.metadata["map_categorical"],
        "n_blocks": model.n_blocks,
        "n_embed": model.n_embed,
        "n_heads": model.n_heads,
        "n_inner": model.n_inner,
        "threshold": thresholds[1] if len(thresholds) > 1 else 0.0,
        "y_names": trn.y_names,
        "mean": trn.metadata["mean"],
        "std": trn.metadata["std"],
    }

    # Save to the tempraroy directory
    torch.save(saved_data, "/tmp/saved_data.pt")

    # Add this artifact to W&B
    artifact = wandb.Artifact("saved_data", type="model")
    artifact.add_file("/tmp/saved_data.pt")
    wandb_run.log_artifact(artifact)

    # Finish the run
    wandb_run.finish()

    # Delete the temporary file
    os.remove("/tmp/saved_data.pt")
