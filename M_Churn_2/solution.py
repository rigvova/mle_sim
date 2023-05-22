from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.utils.extmath import stable_cumsum  # please tell me why I need this


def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float, 
) -> Tuple[float, float]:
    """Calculate the threshold and recall from the Precision-Recall Curve where precision is 
    not less than the provided minimum precision, and recall is max.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_prob : np.ndarray
        Estimated probabilities sorted in descending order.
    min_precision : float
        Minimum required precision.

    Returns
    -------
    float
        The optimal threshold that results in precision not less than `min_precision`.
    float
        The recall associated with the optimal threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    mask = (precisions >= min_precision)
    indices = np.arange(len(recalls))

    max_recall = 0.
    best_i = 0
    for i in indices[mask]:
        if recalls[i] > max_recall:
            max_recall = recalls[i]
            best_i = i
        else:
            break
           
    threshold_proba = thresholds[best_i]

    return threshold_proba, max_recall


def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Tuple[float, float]:
    """Calculate the threshold and recall from the Specificity-Recall Curve where specificity is 
    not less than the provided minimum precision, and recall is max.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_prob : np.ndarray
        Estimated probabilities sorted in descending order.
    min_specificity : float
        Minimum required specificity.

    Returns
    -------
    float
        The optimal threshold that results in specificity not less than `min_specificity`.
    float
        The recall associated with the optimal threshold.
    """
    # True Positives
    tps = stable_cumsum(y_true)
    # False Positives
    fps = np.arange(len(y_true)) + 1 - tps

    thresholds = y_prob.copy()

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]
    fprs = fps / fps[-1]
    recalls = tps / tps[-1]

    best_index = np.searchsorted(
            fprs,  # unlike specificity, it is sorted in asc order
            1 - min_specificity,
            side='right',  # <= 1 - min_specificity
    ) - 1  # we do not want to insert position, but to find it

    threshold_proba = thresholds[best_index]
    max_recall = recalls[best_index]

    return threshold_proba, max_recall


def precision_recall_from_sorted(y_true):
    """
    Calculates precision and recall values from an array of ground truth labels.

    Parameters
    ----------
    y_true : ndarray
        Array of ground truth labels where 1 represents a positive label
        and 0 represents a negative label. Requires sorting, see notes.

    Returns
    -------
    precision : ndarray
        An array of precision values, representing the precision at
        each position in the input array. The first value is always 0.

    recall : ndarray
        An array of recall values, representing the recall at each
        position in the input array. The first value is always 0.

    Notes
    -----
    This function assumes that the input array, `y_true`, is sorted in such a
    way that teh labels correspond to pre-sorted model predictions in descending order.

    If the input array does not contain any positive labels, the recall is
    set to 1 for all thresholds.
    """
    # True Positives
    tps = stable_cumsum(y_true)
    # False Positives
    fps = np.arange(len(y_true)) + 1 - tps
    # Positives
    ps = tps + fps

    # Initialize the result array with zeros to make sure that precision[ps == 0]
    # does not contain uninitialized values.
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # to make sure that the curve starts at (0, 1)
    precision = np.r_[1, precision]
    recall = np.r_[0, recall]

    return precision, recall


def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and its (LCB, UCB)"""
    precision, recall = precision_recall_from_sorted(y_true)

    samples_per_boot = len(y_true)

    # Generate random a matrix of indices for bootstrap
    # index_matrix.shape[0] == n_bootstrap
    index_matrix = np.random.randint(
        0,
        samples_per_boot,
        size=(n_bootstrap, samples_per_boot),
    )

    precisions = np.zeros((n_bootstrap, len(precision)))
    for i, index_list in enumerate(index_matrix):
        sampled_y_true = y_true[index_list]
        sampled_y_prob = y_prob[index_list]
        if np.any(sampled_y_true != sampled_y_true[0]):
            # If there is at least 2 different classes in the sample
            desc_prob_indices = np.argsort(sampled_y_prob, kind="mergesort")[::-1]
            sampled_y_true = sampled_y_true[desc_prob_indices]
            sample_p, sample_r = precision_recall_from_sorted(sampled_y_true)

            # Interpolate values to keep recalls the same
            p_interp = np.interp(recall, sample_r, sample_p)  # sample_r has to be increasing (see np doc)
            precisions[i] = p_interp

        else:
            # If random choice only left one class in the sample
            # we use the original distribution
            precisions[i] = precision

    alpha = (1 - conf) / 2
    precision_lcb = np.percentile(precisions, alpha * 100, axis=0)
    precision_ucb = np.percentile(precisions, (1-alpha) * 100, axis=0)

    return recall, precision, precision_lcb, precision_ucb


def specificity_recall_from_sorted(y_true):
    """
    Calculates precision and recall values from an array of ground truth labels.

    Parameters
    ----------
    y_true : ndarray
        Array of ground truth labels where 1 represents a positive label
        and 0 represents a negative label. Requires sorting, see notes.

    Returns
    -------
    specificity : ndarray
        An array of specificity values, representing the specificity at
        each position in the input array. The first value is always 0.

    recall : ndarray
        An array of recall values, representing the recall at each
        position in the input array. The first value is always 0.

    Notes
    -----
    This function assumes that the input array, `y_true`, is sorted in such a
    way that teh labels correspond to pre-sorted model predictions in descending order.
    """
    # True Positives
    tps = np.cumsum(y_true)
    # False Positives
    fps = np.arange(len(y_true)) + 1 - tps

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    fpr = fps / fps[-1]

    specificity = 1 - fpr
    recall = tps / tps[-1]

    return specificity, recall


def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and its (LCB, UCB)"""
    specificity, recall = specificity_recall_from_sorted(y_true)

    samples_per_boot = len(y_true)

    # Generate random a matrix of indices for bootstrap
    # index_matrix.shape[0] == n_bootstrap
    index_matrix = np.random.randint(
        0,
        samples_per_boot,
        size=(n_bootstrap, samples_per_boot),
    )

    specificities = np.zeros((n_bootstrap, len(specificity)))
    for i, index_list in enumerate(index_matrix):
        sampled_y_true = y_true[index_list]
        sampled_y_prob = y_prob[index_list]
        if np.any(sampled_y_true != sampled_y_true[0]):
            # If there is at least 2 different classes in the sample
            desc_prob_indices = np.argsort(sampled_y_prob, kind="mergesort")[::-1]
            sampled_y_true = sampled_y_true[desc_prob_indices]
            sample_spc, sample_r = specificity_recall_from_sorted(sampled_y_true)

            # Interpolate values to keep recalls the same
            spc_interp = np.interp(recall, sample_r, sample_spc)  # sample_r has to be increasing (see np doc)
            specificities[i] = spc_interp

        else:
            # If random choice only left one class in the sample
            # we use the original distribution
            specificities[i] = specificity

    alpha = (1 - conf) / 2
    specificity_lcb = np.percentile(specificities, alpha * 100, axis=0)
    specificity_ucb = np.percentile(specificities, (1-alpha) * 100, axis=0)

    return recall, specificity, specificity_lcb, specificity_ucb
