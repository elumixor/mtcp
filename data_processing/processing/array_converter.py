from __future__ import annotations

import awkward as ak
import numpy as np
from typing import NamedTuple


Data = NamedTuple("Data", [
    ("x", np.ndarray),
    ("y", list[str] | None),
    ("w", np.ndarray),
    ("selected", np.ndarray),
    ("x_names", list[str]),
    ("y_names", list[str] | None),
    ("event_numbers", np.ndarray),
])


def convert_object_features(data, object_features: list[str], non_object_features: list[str], n_array_elements: int):
    object_features_data = {}

    for feature in object_features:
        feature_data = data[feature]
        max_length = max(ak.num(feature_data))

        # Make sure we have exactly the same number of objects in each event
        max_length = max(max_length, n_array_elements)
        object_feature = ak.to_numpy(ak.pad_none(feature_data, max_length))
        object_feature = object_feature[:, :n_array_elements]

        # Change the names with the indices
        for i in range(n_array_elements):
            object_features_data[f"{feature}_{i}"] = object_feature[:, i]

    w = data["weight"].to_numpy()
    event_numbers = data["eventNumber"].to_numpy()

    n_features_non_object = len(non_object_features)
    n_samples = w.shape[0]
    n_features = n_features_non_object + n_array_elements * len(object_features)

    x = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, feature in enumerate(non_object_features):
        feature_data = data[feature].to_numpy().astype(np.float32)
        x[:, i] = feature_data

    for i, feature in enumerate(object_features_data.values()):
        x[:, n_features_non_object + i] = feature.data

    x_names = non_object_features + list(object_features_data.keys())
    selected = data["selected"].to_numpy().astype(bool)

    return Data(x=x, w=w, selected=selected, x_names=x_names, y=None, y_names=None, event_numbers=event_numbers)
