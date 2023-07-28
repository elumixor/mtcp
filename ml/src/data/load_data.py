import os
import numpy as np
import torch
from tqdm import tqdm

from .data import Data


def load_data(path: str):
    with tqdm(total=12, desc="Loading data") as pbar:
        x_continuous = np.load(os.path.join(path, "x_continuous.npy"))
        pbar.update()

        x_categorical = np.load(os.path.join(path, "x_categorical.npy"))
        pbar.update()

        y = np.load(os.path.join(path, "y.npy"))
        pbar.update()

        w = np.load(os.path.join(path, "w.npy"))
        pbar.update()

        mean = np.load(os.path.join(path, "mean.npy"))
        pbar.update()

        std = np.load(os.path.join(path, "std.npy"))
        pbar.update()

        selected = np.load(os.path.join(path, "selected.npy"))
        pbar.update()

        x_names_categorical = np.load(os.path.join(path, "x_names_categorical.npy"))
        pbar.update()

        x_names_continuous = np.load(os.path.join(path, "x_names_continuous.npy"))
        pbar.update()

        y_names = np.load(os.path.join(path, "y_names.npy"))
        pbar.update()

        map_categorical = np.load(os.path.join(path, "map_categorical.npy"), allow_pickle=True).item()
        pbar.update()

        event_numbers = np.load(os.path.join(path, "event_numbers.npy"))
        pbar.update()


    # Split into selected and not selected
    return Data(
        x_continuous=torch.from_numpy(x_continuous),
        x_categorical=torch.from_numpy(x_categorical),
        y=torch.from_numpy(y),
        w=torch.from_numpy(w),
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
        x_names_categorical=x_names_categorical.tolist(),
        x_names_continuous=x_names_continuous.tolist(),
        y_names=y_names.tolist(),
        map_categorical=map_categorical,
        event_numbers=event_numbers.tolist(),
    ), torch.from_numpy(selected)
