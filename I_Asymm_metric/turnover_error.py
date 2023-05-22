import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """Returns asymmetric loss that punishes for missed profit."""
    diff = y_true - y_pred
    denominator = np.abs(y_pred) + 1  # we need to add 1 in denominator for cases when y_pred==0
    loss = np.sum(
        (diff / denominator) ** 2
    )

    return loss
