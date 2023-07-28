import torch
from .significance import significance
from .confusion_matrix import confusion_matrix


def find_significance_threshold(
    probs: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: torch.FloatTensor,
    signal_idx: int,
    F: float,
    n_significance_thresholds=100,
    return_stats=False,
    include_signal=True,
    min_background=float("-inf"),
):
    thresholds = torch.linspace(0, 1, n_significance_thresholds)

    best_threshold = None
    best_S = float("-inf")

    if return_stats:
        stats = []

    for threshold in thresholds:
        passed = probs[:, signal_idx] >= threshold
        probs_ = probs.clone()
        probs_[:, signal_idx] = -1
        best_bg = probs_.argmax(dim=1)
        pred = torch.where(passed, signal_idx, best_bg)

        cm = confusion_matrix(pred, targets, weights, signal=signal_idx)
        S = significance(cm, F, include_signal=include_signal, min_background=min_background)

        if return_stats:
            tp = cm[0, 0] * F
            fp = cm[1, 0] * F
            stats.append((threshold, S, tp, fp))

        if S >= best_S:
            best_threshold = threshold
            best_S = S

    if not return_stats:
        return best_threshold, best_S

    return best_threshold, best_S, stats
