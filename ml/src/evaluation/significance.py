import torch


def significance(confusion_matrix: torch.FloatTensor, F=1, include_signal=True, min_background=float("-inf")):
    assert confusion_matrix.shape == (2, 2), f"Confusion matrix must be 2x2, got {confusion_matrix.shape}"

    signal = confusion_matrix[0, 0]  # TP
    background = confusion_matrix[1, 0]  # FP

    # With weights, sometimes we can have either negative or zero
    if signal + background <= 0 or background < min_background:
        significance = torch.tensor(0.0)
    else:
        denom = background if not include_signal else (background + signal)
        if denom <= 0:
            return torch.tensor(0.0)

        significance = (signal / (denom ** 0.5)) * (F ** 0.5)

    return significance
