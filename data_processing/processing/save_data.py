import os
import numpy as np

from .process_data import ProcessedData


def save_data(data: ProcessedData, path: str):
    path = os.path.join("data_processing", path)

    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, "x_categorical.npy"), data.x_categorical)
    np.save(os.path.join(path, "x_continuous.npy"), data.x_continuous)
    np.save(os.path.join(path, "y.npy"), data.y)
    np.save(os.path.join(path, "w.npy"), data.w)
    np.save(os.path.join(path, "mean.npy"), data.mean)
    np.save(os.path.join(path, "std.npy"), data.std)
    np.save(os.path.join(path, "selected.npy"), data.selected)
    np.save(os.path.join(path, "x_names_categorical.npy"), data.x_names_categorical)
    np.save(os.path.join(path, "x_names_continuous.npy"), data.x_names_continuous)
    np.save(os.path.join(path, "y_names.npy"), data.y_names)
    np.save(os.path.join(path, "map_categorical.npy"), data.map_categorical)
    np.save(os.path.join(path, "event_numbers.npy"), data.event_numbers)
