from __future__ import annotations

from collections import defaultdict
import numpy as np
from tqdm import tqdm
from typing import NamedTuple

from .array_converter import Data


ProcessedData = NamedTuple("ProcessedData", [
    ("x_categorical", np.ndarray),
    ("x_continuous", np.ndarray),
    ("y", list[str] | None),
    ("w", np.ndarray),
    ("mean", np.ndarray),
    ("std", np.ndarray),
    ("selected", np.ndarray),
    ("x_names_categorical", list[str]),
    ("x_names_continuous", list[str]),
    ("y_names", list[str] | None),
    ("map_categorical", dict[str, dict[int, str]] | None),
    ("event_numbers", np.ndarray),
])


def process_data(data: Data, categorical: list[str], invalid: dict[int, list[str]]):
    # Split x into categorical and continuous
    i_categorical = [data.x_names.index(feature) for feature in categorical]
    i_continuous = [i for i in range(len(data.x_names)) if i not in i_categorical]

    print("Casting categorical features to int")
    x_categorical = data.x[:, i_categorical].astype(int)
    x_continuous = data.x[:, i_continuous]

    x_names_categorical = [data.x_names[i] for i in i_categorical]
    x_names_continuous = [data.x_names[i] for i in i_continuous]

    for value, features in tqdm(invalid.items(), desc="Setting invalid values to NaN"):
        for feature in features:
            # For invalid values in continuous features, simply replace with NaN
            try:
                i = x_names_continuous.index(feature)
                x_continuous[x_continuous[:, i] == value, i] = np.nan
            except ValueError:
                pass

    # Remap categorical values to the [0, 1, ..., n_categories - 1] range
    map_categorical = defaultdict(dict)
    for i, feature_name in enumerate(tqdm(x_names_categorical, desc="Remapping categorical values")):
        features = x_categorical[:, i]

        # First we need to shift by the minimum value to avoid any collisions
        min_value = np.min(features)
        features -= min_value

        # Next check all the values from 0 to max
        max_value = np.max(features)

        new_index = 0
        for j in range(max_value + 1):
            matches = features == j

            if matches.sum() > 0:
                map_categorical[feature_name][new_index] = f"{feature_name}={j + min_value}"
                features[matches] = new_index
                new_index += 1

    # Find the mean and std of the continuous features
    mean = np.nanmean(x_continuous, axis=0)
    std = np.nanstd(x_continuous, axis=0)

    # Drop the continuous features with std = 0
    i_valid = std != 0

    mean = mean[i_valid]
    std = std[i_valid]
    x_continuous = (x_continuous[:, i_valid] - mean) / std
    x_names_continuous = [name for i, name in enumerate(x_names_continuous) if i_valid[i]]

    assert len(x_names_continuous) == len(mean) == len(std) == x_continuous.shape[1]
    assert len(x_names_categorical) == x_categorical.shape[1]

    return ProcessedData(
        x_categorical=x_categorical,
        x_continuous=x_continuous,
        y=data.y,
        w=data.w,
        mean=mean,
        std=std,
        selected=data.selected,
        x_names_categorical=x_names_categorical,
        x_names_continuous=x_names_continuous,
        y_names=data.y_names,
        map_categorical=map_categorical,
        event_numbers=data.event_numbers,
    )
