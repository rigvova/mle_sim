from typing import List

import numpy as np


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method == 'standard':
        denominator = np.log2(np.arange(k)+2)

        dcg = np.sum(relevance[:k] / denominator)
        idcg = np.sum(sorted(relevance, reverse=True)[:k] / denominator)

        score = dcg / idcg

    elif method == 'industry':
        denominator = np.log2(np.arange(k)+2)

        dcg = np.sum((np.power(2, relevance[:k]) - 1) / denominator)
        idcg = np.sum((np.power(2, sorted(relevance, reverse=True)[:k]) - 1) / denominator)

        score = dcg / idcg

    else:
        raise ValueError

    return score


def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """Average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method in ['standard', 'industry']:
        n = len(list_relevances)
        s = 0  # sum
        for query_list in list_relevances:
            s += normalized_dcg(query_list, k, method)
        score = s / n

    else:
        raise ValueError

    return score
