import yaml
import os

def read_yaml(file_path: str):
    if file_path.endswith(".yaml"):
        file_path = file_path[:-5]

    file_path = os.path.join("data_processing", f"{file_path}.yaml")
    with open(file_path, "r") as f:
        return yaml.safe_load(f)