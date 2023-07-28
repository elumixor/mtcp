import torch


def confusion_matrix(
        preds: torch.LongTensor,
        target: torch.LongTensor,
        weights: torch.FloatTensor,
        n_classes: int | None = None,
        signal: int | None = None,
) -> torch.FloatTensor:
    assert (signal is None) + (n_classes is None) == 1, f"Either specify signal for binary classification or n_classes for multiclass classification"
    assert preds.shape == target.shape == weights.shape, f"{preds.shape} != {target.shape} != {weights.shape}"
    assert n_classes is None or preds.max() < n_classes, f"{preds.max()} > {n_classes - 1}"
    assert n_classes is None or target.max() < n_classes, f"{target.max()} > {n_classes - 1}"
    assert target.device == preds.device == weights.device, f"{target.device} != {preds.device} != {weights.device}"

    # If signal is specified, merge all background classes into one
    if signal is not None:
        preds = preds != signal  # We use != to have the correct prediction as the 0 class
        target = target != signal
        n_classes = 2

    result = torch.empty((n_classes, n_classes), dtype=torch.float32, device=preds.device)
    for i_true in range(n_classes):
        for i_pred in range(n_classes):
            mask = (target == i_true) & (preds == i_pred)
            result[i_true, i_pred] = weights[mask].sum()

    return result

def make_binary_cm(cm: torch.FloatTensor, signal_idx: int):
    cm_binary = torch.zeros((2, 2))
    cm_binary[0, 0] = cm[signal_idx, signal_idx]
    cm_binary[1, 0] = cm[:, signal_idx].sum() - cm_binary[0, 0]
    cm_binary[0, 1] = cm[signal_idx].sum() - cm[signal_idx, signal_idx]
    cm_binary[1, 1] = cm.sum() - cm_binary.sum()
    return cm_binary