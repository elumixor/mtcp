import os
import numpy as np

from .process_data import ProcessedData


def load_data(path: str):
    x_categorical = np.load(os.path.join(path, "x_categorical.npy"))
    x_continuous = np.load(os.path.join(path, "x_continuous.npy"))
    y = np.load(os.path.join(path, "y.npy"))
    w = np.load(os.path.join(path, "w.npy"))
    mean = np.load(os.path.join(path, "mean.npy"))
    std = np.load(os.path.join(path, "std.npy"))
    selected = np.load(os.path.join(path, "selected.npy"))
    x_names_categorical = np.load(os.path.join(path, "x_names_categorical.npy"))
    x_names_continuous = np.load(os.path.join(path, "x_names_continuous.npy"))
    y_names = np.load(os.path.join(path, "y_names.npy"))
    map_categorical = np.load(os.path.join(path, "map_categorical.npy"), allow_pickle=True).item()
    event_numbers = np.load(os.path.join(path, "event_numbers.npy"))

    return ProcessedData(
        x_categorical=x_categorical,
        x_continuous=x_continuous,
        y=y,
        w=w,
        mean=mean,
        std=std,
        selected=selected,
        x_names_categorical=x_names_categorical,
        x_names_continuous=x_names_continuous,
        y_names=y_names,
        map_categorical=map_categorical,
        event_numbers=event_numbers
    )
