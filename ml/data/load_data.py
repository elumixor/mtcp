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


def load_from_config(config):
    data, selected = load_data(config.data_path)
    data_cut = data[selected]
    data_uncut = data[~selected]

    classes = config.classes if "classes" in config else data.y_names
    features = config.features if "features" in config else data.x_names

    # Take the trn_split of the selected samples
    # 20% of that is the validation set
    trn_cut, val, tst = data_cut.split(config.trn_split)
    trn_uncut = trn_cut + data_uncut

    # Here we determine the training set we're using
    assert config.cuts in ["apply", "discard"]
    trn = trn_uncut if config.cuts == "discard" else trn_cut

    # Also apply the fraction cut if needed
    if config.fraction < 1:
        n_samples = int(config.fraction * trn.n_samples)
        # Shuffle the data
        indices = torch.randperm(trn.n_samples)[:n_samples]
        trn = trn[indices]

    print()
    print(f"Training set: {trn.n_samples} samples")
    print(f"Validation set: {val.n_samples} samples")
    print()

    # Select classes
    trn = trn.select_classes(classes)
    val = val.select_classes(classes)
    tst = tst.select_classes(classes)

    using_all_features = len(features) == (
        len(data.x_names_categorical) + len(data.x_names_continuous)
    )
    print(f"Using {len(features)} features{' (all)' if using_all_features else ''}")
    print()

    # Select features
    trn = trn.select_features(features)
    val = val.select_features(features)
    tst = tst.select_features(features)

    # Merge all the background classes (everything excepr ttH) into one
    if config.use_binary:
        background_classes = [c for c in trn.y_names if c != "ttH"]
        trn = trn.merge_classes(names=background_classes, new_class_name="background")
        val = val.merge_classes(names=background_classes, new_class_name="background")
        tst = tst.merge_classes(names=background_classes, new_class_name="background")

    return trn, val, tst
