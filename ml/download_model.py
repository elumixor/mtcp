import autorootcwd  # Do not delete - adds the root of the project to the path

import os
import wandb
import re

from ml.utils import get_config
from ml.training import Checkpoint


def get_run(run_name: str, project_name="mtcp"):
    # Get the artifacts of the run
    api = wandb.Api()

    # Find the runs matching the name
    runs = api.runs(project_name, {"$and": [{"state": "finished"}]})
    for run in runs:
        if run_name == run.name:
            break
    else:
        raise ValueError(f"Run {run_name} not found")

    print(f"Found run {run.name} ({run.id})")

    return run


def download_model_artifact(run, clean=False):
    file_name = f"ml/artifacts/{run.name}/model-full.pt"

    # Return the path to the model file if it exists
    if os.path.exists(file_name):
        if not clean:
            return file_name

        os.remove(file_name)

    # Get the model artifact
    for artifact in run.logged_artifacts():
        if artifact.type == "model":
            break
    else:
        raise ValueError(f"Model artifact not found")

    print(f"Found model artifact {artifact.type} {artifact.name} ({artifact.id})")

    # Download the model
    model_artifact_dir = artifact.download(root=f"ml/artifacts/{run.name}")

    # Rename it to "model.pt"
    os.rename(f"{model_artifact_dir}/{artifact.files()[0].name}", file_name)

    # Get the model file
    print(f"Model artifact downloaded to {file_name}")

    return file_name


def download_checkpoint_artifact(run, clean=False, best=True):
    checkpoint_name = f"checkpoint-{'best' if best else 'last'}"
    checkpoint_artifact_dir = f"ml/artifacts/{run.name}"

    model_path = f"{checkpoint_artifact_dir}/model.pt"
    optim_path = f"{checkpoint_artifact_dir}/optim.pt"
    scheduler_path = f"{checkpoint_artifact_dir}/scheduler.pt"

    if os.path.exists(model_path) and os.path.exists(optim_path) and os.path.exists(scheduler_path):
        if not clean:
            return Checkpoint(model=model_path, optim=optim_path, scheduler=scheduler_path, stats=None)

        os.remove(model_path)
        os.remove(optim_path)
        os.remove(scheduler_path)

    # Get the checkpoint artifact
    for artifact in run.logged_artifacts():
        # Check if artiface name matches
        if re.match(f"{checkpoint_name}.*", artifact.name):
            break
    else:
        raise ValueError(f"{checkpoint_name} artifact not found")

    print(f"Found checkpoint artifact {artifact.type} {artifact.name} ({artifact.id})")

    # Download the checkpoint files
    checkpoint_artifact_dir = artifact.download(root=f"ml/artifacts/{run.name}")

    return Checkpoint(model=model_path, optim=optim_path, scheduler=scheduler_path, stats=None)


def download_model(config: str, clean=False, checkpoint=False, silent=False):
    config = get_config(config, check_cuda_device=False, silent=silent)
    project_name = config.project_name
    run_name = config.run_name

    run = get_run(run_name, project_name)

    model_path = download_model_artifact(run, clean=clean)

    if not checkpoint:
        return model_path

    checkpoint_path = download_checkpoint_artifact(run, clean=clean)

    return model_path, checkpoint_path


if __name__ == "__main__":
    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Download the trained model")
    parser.add_argument("config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--clean", action="store_true", help="Clean the model directory")
    args = parser.parse_args()

    # Read the config
    model = download_model(args.config, clean=args.clean)
