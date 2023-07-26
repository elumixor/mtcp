
from __future__ import annotations

from tqdm import tqdm
import uproot
import awkward as ak
import numpy as np

from .array_converter import convert_object_features, Data
from .config_parser import ConfigParser


def read_region(region: str,
                parser: ConfigParser,
                train_features: list[str],
                object_features: list[str],
                samples=None,
                nested_size=6):

    if samples is None:
        samples = parser.samples

    non_object_features = list(set(train_features) - set(object_features))

    process_data = {}
    for s in samples:
        files = [f"{file}:{parser.ntuple_name}" for file in parser.files_by_process()[s]]
        weight = parser.weight_expr(process=s, with_luminosity=True)
        cut_expr = parser.cut_expr(region, s)

        concatenated = None

        # With uproot.concatenate this would have been more elegant, but it fails on files with no events
        # This we barbarically iterate over the files in a simple for loop
        for file in tqdm(files, desc=s):
            with uproot.open(file) as f:
                if len(f) == 0:
                    continue

                arrays = f.arrays([*train_features, "weight", "selected", "eventNumber"],
                                  cut=None, aliases={"weight": weight, "selected": cut_expr})

                if len(arrays) == 0:
                    continue

                # Concatenate with the previous arrays
                concatenated = arrays if concatenated is None else ak.concatenate([concatenated, arrays], axis=0)

        # Convert to numpy, process object features to masked arrays
        if concatenated is not None:
            process_data[s] = convert_object_features(concatenated, object_features, non_object_features, nested_size)

    # Concatenate all the processes
    print(f"Concatenating data from {len(process_data)} processes")

    y_names = list(process_data.keys())
    x_names = process_data[y_names[0]].x_names

    x = np.concatenate([process_data[y_name].x for y_name in y_names])
    y = np.concatenate([np.full(process_data[y_name].x.shape[0], i) for i, y_name in enumerate(y_names)])
    w = np.concatenate([process_data[y_name].w for y_name in y_names])
    event_numbers = np.concatenate([process_data[y_name].event_numbers for y_name in y_names])

    print(x_names)

    selected = np.concatenate([process_data[y_name].selected for y_name in y_names])

    return Data(
        x=x,
        w=w,
        y=y,
        selected=selected,
        x_names=x_names,
        y_names=y_names,
        event_numbers=event_numbers,
    )
