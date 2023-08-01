import autorootcwd  # Do not delete - adds the root of the project to the path

import wandb

from ml.utils import get_config

# Download the model from wandb
def get_model(run_name: str):
    run = wandb.init(run_name)

if __name__ == "__main__":
    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Read the config
    config = get_config(args.config)

    model = get_model(config.run_name)