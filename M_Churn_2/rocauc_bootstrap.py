from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Computes confidence bounds of the ROC-AUC.

    Parameters
    ----------
    classifier : `ClassifierMixin`
        A pre-trained sklearn classifier.
    X : `np.ndarray`
        Input features.
    y : `np.ndarray`
        Input labels.
    conf : `float`, optional
        Confidence interval size. Defualts to 0.95.
    n_bootstraps : `int`, optional
        Number of nootstraps to perform. Defualts to 10000.

    Returns
    -------
    Tuple[float, float]
        Lower and upper confidence bounds.
    """
    if np.all(y == y[0]):
        # All labels are the same, ROC-AUC is undefined
        return (0., 1.)
    
    reference_auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])

    n_samples = min(1000, X.shape[0])

    # Generate random a matrix of indices for bootstrap
    # index_matrix.shape[0] == n_bootstraps
    index_matrix = np.random.randint(
        0,
        X.shape[0], 
        size=(n_bootstraps, n_samples),
    )

    # Make bootstrapped data samples
    boot_X = np.zeros(shape=(n_bootstraps, n_samples, X.shape[1]))
    boot_y = np.zeros(shape=(n_bootstraps, n_samples), dtype=np.int32)

    for i in range(n_bootstraps):
        boot_X[i] = X[index_matrix[i]]
        boot_y[i] = y[index_matrix[i]]

    # Make predictions for all data at once
    preds = (classifier.predict_proba(
        boot_X.reshape(n_bootstraps*n_samples, X.shape[1]))
    )[:, 1]

    preds = preds.reshape(n_bootstraps, n_samples)

    auc_scores = []
    for i in range(index_matrix.shape[0]):
        if np.any(boot_y[i] != boot_y[i][0]):
            # If there is at least 2 different classes in the sample
            auc_scores.append(roc_auc_score(boot_y[i], preds[i]))

        else:
            # If random choice only left one class in the sample
            # we use the original distribution
            auc_scores.append(reference_auc)

    alpha = (1 - conf) / 2
    lcb = np.percentile(auc_scores, alpha * 100)
    ucb = np.percentile(auc_scores, (1-alpha) * 100)

    return lcb, ucb
