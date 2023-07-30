from __future__ import annotations
import torch
import math


class Data:
    def __init__(self,
                 x_continuous: torch.FloatTensor,
                 x_categorical: torch.LongTensor,
                 y: torch.LongTensor,
                 w: torch.FloatTensor,
                 **metadata):
        self.x_continuous = x_continuous
        self.x_categorical = x_categorical
        self.y = y
        self.w = w
        self.metadata = metadata

    @property
    def y_names(self):
        return self.metadata["y_names"] if "y_names" in self.metadata else None

    @property
    def x_names_continuous(self):
        return self.metadata["x_names_continuous"] if "x_names_continuous" in self.metadata else None

    @property
    def x_names_categorical(self):
        return self.metadata["x_names_categorical"] if "x_names_categorical" in self.metadata else None

    @property
    def x_names(self):
        x_names_continuous = self.x_names_continuous
        x_names_categorical = self.x_names_categorical

        if x_names_continuous is None or x_names_categorical is None:
            return None

        return x_names_categorical + x_names_continuous

    @property
    def n_samples(self):
        return self.x_continuous.shape[0]

    @property
    def n_features(self):
        return self.x_continuous.shape[1] + self.x_categorical.shape[1]

    @property
    def n_features_continuous(self):
        return self.x_continuous.shape[1]

    @property
    def categorical_sizes(self):
        return (self.x_categorical.max(dim=0)[0] + 1).tolist()

    @property
    def n_classes(self):
        class_names = self.y_names
        return len(class_names) if class_names is not None else self.y.max() + 1

    @property
    def n_samples_weighted(self):
        return self.w.sum()

    @property
    def class_weights(self):
        """
        Weighting for each class to balance the dataset
        Calcualted as `weight / n_samples` per each class where `weight` and `n_samples` are normalized over classes
        """
        class_weights = torch.empty(self.n_classes)
        class_n_samples = torch.empty(self.n_classes)
        for i in range(self.n_classes):
            selected = self.y == i
            class_n_samples[i] = selected.sum()
            class_weights[i] = self.w[selected].sum()

        # Normalize
        # class_weights /= class_weights.sum()
        # class_n_samples /= class_n_samples.sum()

        # Calculate the weights
        result = class_weights / class_n_samples
        # result = class_weights
        # result = class_n_samples.sum() / class_n_samples

        # We could have NaNs or Infs if there are classes with no samples
        result[torch.isnan(result)] = 0
        result[torch.isinf(result)] = 0

        # Normalize the weights
        result /= result.sum()

        return result

    @classmethod
    def zeros(cls, n_features_continues: int, categorical_sizes: list[int], n_samples=1, **metadata):
        x_continuous = torch.zeros(n_samples, n_features_continues)
        x_categorical = torch.zeros(n_samples, len(categorical_sizes), dtype=torch.long)
        y = torch.zeros(n_samples, dtype=torch.long)
        w = torch.ones(n_samples)

        return cls(x_continuous, x_categorical, y, w, **metadata)

    def __getitem__(self, i):
        return Data(self.x_continuous[i], self.x_categorical[i], self.y[i], self.w[i], **self.metadata)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Data(n_samples={self.n_samples}, n_features={self.n_features}, n_classes={self.n_classes})"

    def __add__(self, other):
        assert isinstance(other, Data), f"Expected Data, got {type(other)}"
        assert self.y_names == other.y_names, f"Expected same class names, got {self.y_names} and {other.y_names}"
        assert self.x_names == other.x_names, f"Expected same feature names, got {self.x_names} and {other.x_names}"

        x_continuous = torch.cat([self.x_continuous, other.x_continuous])
        x_categorical = torch.cat([self.x_categorical, other.x_categorical])
        y = torch.cat([self.y, other.y])
        w = torch.cat([self.w, other.w])

        return Data(x_continuous, x_categorical, y, w, **self.metadata)

    def __eq__(self, other):
        return isinstance(other, Data) and \
            self.y_names == other.y_names and \
            self.x_names == other.x_names and \
            self.n_samples == other.n_samples and \
            torch.allclose(self.x_continuous, other.x_continuous, equal_nan=True) and \
            torch.all(self.x_categorical == other.x_categorical) and \
            torch.all(self.y == other.y) and \
            torch.allclose(self.w, other.w)

    def batches(self, batch_size=64, shuffle=True):
        """
        Iterate with the batches
        """
        class batch_iterator:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                indices = torch.randperm(self.data.n_samples) if shuffle else torch.arange(self.data.n_samples)
                start = 0
                while True:
                    end = start + batch_size
                    yield self.data[indices[start:end]]  # Yield the batch
                    if end >= self.data.n_samples:
                        break
                    start = end

            def __len__(self):
                return math.ceil(self.data.n_samples / batch_size)

        return batch_iterator(self)

    def to(self, device):
        x_continuous = self.x_continuous.to(device)
        x_categorical = self.x_categorical.to(device)
        y = self.y.to(device)
        w = self.w.to(device)

        return Data(x_continuous, x_categorical, y, w, **self.metadata)

    def merge_classes(self, names=None, indices=None, new_class_name="merged"):
        """Merges the specified classes into a single class. Returns the new Data object"""
        assert (names is None) != (indices is None), "Either names or indices must be specified"

        if indices is None:
            indices = [self.y_names.index(name) for name in names]

        kept_indices = [i for i in range(self.n_classes) if i not in indices]

        y = self.y.clone()

        for new_i, i in enumerate(kept_indices):
            y[self.y == i] = new_i  # Note that we use the old labels to determine the samples, which don't get modified

        for i in indices:
            y[self.y == i] = len(kept_indices)

        y_names = [self.y_names[i] for i in kept_indices] + [new_class_name]

        metadata = self.metadata.copy()
        metadata["y_names"] = y_names

        return Data(self.x_continuous, self.x_categorical, y, self.w, **metadata)

    def select_classes(self, names=None, indices=None):
        assert (names is None) != (indices is None), "Either names or indices must be specified"

        if indices is None:
            indices = [self.y_names.index(name) for name in names]

        # Sort the indices so that we can use them to index the classes
        indices = sorted(indices)

        selected = torch.zeros(self.n_samples, dtype=torch.bool)

        for i in indices:
            selected[self.y == i] = True

        x_continuous_selected = self.x_continuous[selected]
        x_categorical_selected = self.x_categorical[selected]
        w_selected = self.w[selected]
        y_selected = self.y[selected]

        # Renumber the classes
        y = torch.zeros_like(y_selected)
        for new_i, i in enumerate(indices):
            y[y_selected == i] = new_i

        y_names = [self.y_names[i] for i in indices]

        return Data(x_continuous_selected, x_categorical_selected, y, w_selected, **{
            **self.metadata,
            "y_names": y_names
        })

    def select_features(self, names=None, indices=None):
        assert (names is None) != (indices is None), "Either names or indices must be specified"

        if names is None:
            names = [self.x_names[i] for i in indices]

        names_continuous = self.metadata["x_names_continuous"]
        names_categorical = self.metadata["x_names_categorical"]

        indices_continuous = [names_continuous.index(name) for name in names if name in names_continuous]
        indices_categorical = [names_categorical.index(name) for name in names if name in names_categorical]

        x_continuous = self.x_continuous[:, indices_continuous]
        x_categorical = self.x_categorical[:, indices_categorical]

        metadata = self.metadata.copy()
        metadata["x_names_continuous"] = [names_continuous[i] for i in indices_continuous]
        metadata["x_names_categorical"] = [names_categorical[i] for i in indices_categorical]

        return Data(x_continuous, x_categorical, self.y, self.w, **metadata)

    def split(self, trn=0.8, val: float | None = None, tst=0.0, return_indices=False):
        if val is None:
            val = 1 - trn - tst

        assert trn + val + tst == 1, "The sum of the fractions must be 1"

        # Randomly shuffle the dataset
        indices = torch.randperm(self.n_samples)

        trn_end = math.floor(trn * self.n_samples)
        val_end = trn_end + math.ceil(val * self.n_samples)

        i_trn = indices[:trn_end]
        i_val = indices[trn_end:val_end]
        i_tst = indices[val_end:]

        trn_data = self[i_trn]
        val_data = self[i_val]
        tst_data = self[i_tst]

        if not return_indices:
            return trn_data, val_data, tst_data

        return trn_data, val_data, tst_data, (i_trn, i_val, i_tst)
