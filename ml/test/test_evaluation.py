import torch

from src.evaluation import confusion_matrix, make_binary_cm


def test_weighted_confusion_matrix():
    preds = torch.tensor([1, 2, 3])
    target = torch.tensor([1, 2, 3])
    weights = torch.tensor([1, 2, 3])
    n_classes = 4

    cm = confusion_matrix(preds, target, weights, n_classes)

    assert cm.shape == (n_classes, n_classes)

    assert (cm == torch.tensor([[0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 2, 0],
                                [0, 0, 0, 3]])).all()

    binary_cm = make_binary_cm(cm, signal_idx=2)
    assert binary_cm.shape == (2, 2)
    assert (binary_cm == torch.tensor([[2, 0],
                                       [0, 4]])).all()


def test_confusion_matrix_with_incorrect_preds():
    y_true = torch.tensor([2, 2, 2])
    y_pred = torch.tensor([1, 2, 3])

    weights = torch.tensor([1, 2, 3])
    n_classes = 4

    cm = confusion_matrix(y_pred, y_true, weights, n_classes)

    assert cm.shape == (n_classes, n_classes)

    assert (cm == torch.tensor([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 1, 2, 3],
                                [0, 0, 0, 0]])).all()

    binary_cm = make_binary_cm(cm, signal_idx=2)
    assert binary_cm.shape == (2, 2)
    assert (binary_cm == torch.tensor([[2, 4],
                                       [0, 0]])).all()


def test_confusion_matrix_with_incorrect_preds_2():
    y_true = torch.tensor([1, 2, 3])
    y_pred = torch.tensor([2, 2, 2])

    weights = torch.tensor([1, 2, 3])
    n_classes = 4

    cm = confusion_matrix(y_pred, y_true, weights, n_classes)

    assert cm.shape == (n_classes, n_classes)

    assert (cm == torch.tensor([[0, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 2, 0],
                                [0, 0, 3, 0]])).all()

    binary_cm = make_binary_cm(cm, signal_idx=2)
    assert binary_cm.shape == (2, 2)
    assert (binary_cm == torch.tensor([[2, 0],
                                       [4, 0]])).all()

def test_confusion_matrix_binary():
    y_true = torch.tensor([1, 2, 3])
    y_pred = torch.tensor([2, 2, 2])

    weights = torch.tensor([1, 2, 3])

    cm = confusion_matrix(y_pred, y_true, weights, signal=2)

    assert cm.shape == (2, 2)
    assert (cm == torch.tensor([[2, 0],
                                [4, 0]])).all()


def test_if_we_classify_as_background_it_does_not_matter_which_one():
    preds = torch.tensor([1, 2, 3])
    target = torch.tensor([1, 2, 3])
    weights = torch.tensor([1, 2, 3])

    cm_1 = confusion_matrix(preds, target, weights, signal=2)

    preds = torch.tensor([3, 2, 1])

    cm_2 = confusion_matrix(preds, target, weights, signal=2)

    assert (cm_1 == cm_2).all()
