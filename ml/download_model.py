import autorootcwd  # Do not delete - adds the root of the project to the path

import os
import wandb

from ml.utils import get_config

# Download the model from wandb


def download_model(project_name: str, run_name: str):
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

    # Get the model artifact
    for artifact in run.logged_artifacts():
        if artifact.type == "model":
            break
    else:
        raise ValueError(f"Model artifact not found")

    print(f"Found model artifact {artifact.type} {artifact.name} ({artifact.id})")

    # Download the model
    model_artifact_dir = artifact.download(root=f"ml/artifacts/{run_name}")

    # Rename it to "model.pt"
    file_name = f"{model_artifact_dir}/model.pt"
    os.rename(f"{model_artifact_dir}/{artifact.files()[0].name}", file_name)

    # Get the model file
    print(f"Model artifact downloaded to {file_name}")

    return file_name


if __name__ == "__main__":
    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Read the config
    config = get_config(args.config, check_cuda_device=False)

    model = download_model(config.project_name, config.run_name)
