from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values​​
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method == 'standard':
        score = np.sum(relevance[:k] / np.log2(np.arange(k)+2))
    elif method == 'industry':
        score = np.sum((np.power(2, relevance[:k]) - 1) / np.log2(np.arange(k) + 2))
    else:
        raise ValueError

    return score
