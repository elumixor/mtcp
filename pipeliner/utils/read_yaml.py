import yaml
from .dotdict import DotDict

def read_yaml(path: str):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    return DotDict.from_dict(config)
