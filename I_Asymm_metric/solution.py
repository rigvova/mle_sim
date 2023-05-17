import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """Returns asymmetric loss that punishes for missed profit."""
    loss = np.sum(
        ((y_true - y_pred) / (np.abs(y_pred)+1)) ** 2
    )

    return loss
