import numpy as np


def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """Returns asymmetric loss that punishes for overestimation."""
    diff = y_true - y_pred
    abs_diff = np.abs(diff)

    w_underestimation = 1.0
    w_overestimation = 10.0

    errors = np.where(
        diff < 0,
        abs_diff * w_overestimation,
        abs_diff * w_underestimation
    )

    amse = np.mean(errors)

    return amse
