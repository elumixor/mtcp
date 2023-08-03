import os
import torch
from pipeliner.utils import read_yaml, DotDict


def get_config(config_path: str, check_cuda_device=True, silent=False):
    # Run name is the file name
    run_name = os.path.splitext(os.path.basename(config_path))[0]

    def get_config(path: str):
        # If the file exists, use it
        if not os.path.exists(path):
            # Check if the file exists in the ml/configs directory
            path = f"./ml/configs/{path}.yaml"
            if not os.path.exists(path):
                raise ValueError(f"Config file {path} not found")

        return read_yaml(path)

    config = DotDict(extend=config_path)

    # Check for the "extend"
    while "extend" in config:
        extend_path = config.extend
        del config.extend

        # Merge the two dictionaries
        base_config = get_config(extend_path)
        for key, value in config.items():
            base_config[key] = value

        config = base_config

    if "run_name" not in config:
        config.run_name = run_name

    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.device != "cuda" and "require_cuda" in config and config.require_cuda:
        raise ValueError("CUDA is not available but is required")

    if config.device == "cuda":
        # Make sure that CUDA env is provided
        if check_cuda_device and "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise ValueError("CUDA_VISIBLE_DEVICES not set")

        config.cuda_device = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else 0

    config.dtype = torch.float16 if config.use_half else torch.float32
    config.compile = config.compile and torch.cuda.get_device_capability()[0] >= 7
    config.checkpoints_dir = config.checkpoints_dir if "checkpoints_dir" in config else \
        os.environ["MTCP_CHECKPOINTS_DIR"] if "MTCP_CHECKPOINTS_DIR" in os.environ else \
        "ml/checkpoints"

    if not silent:
        print(f"\n{' Config ':-^80}\n")
        print(config)
        print(f"\n{'':-^80}\n")

    return config
