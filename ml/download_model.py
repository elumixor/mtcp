import autorootcwd  # Do not delete - adds the root of the project to the path

import os
import wandb

from ml.utils import get_config


def download_model(config: str, clean=False):
    config = get_config(config, check_cuda_device=False)
    project_name = config.project_name
    run_name = config.run_name

    file_name = f"ml/artifacts/{run_name}/model.pt"

    # Return the path to the model file if it exists
    if os.path.exists(file_name):
        if not clean:
            return file_name

        os.remove(file_name)

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
    os.rename(f"{model_artifact_dir}/{artifact.files()[0].name}", file_name)

    # Get the model file
    print(f"Model artifact downloaded to {file_name}")

    return file_name


if __name__ == "__main__":
    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Download the trained model")
    parser.add_argument("config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--clean", action="store_true", help="Clean the model directory")
    args = parser.parse_args()

    # Read the config
    model = download_model(args.config, clean=args.clean)
