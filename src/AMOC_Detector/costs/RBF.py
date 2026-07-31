"""
A kernel based method for detecting changepoints 


"""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel


def rbf_kernel(time_series: NDArray) -> NDArray:
    """
    Kernel based reduction in cost from adding one changepoint
    """
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    n = len(x)
    out = np.zeros(n)

    length_scale = _median_heuristic(x)
    gram = sklearn_rbf_kernel(x, gamma=1 / (2 * length_scale**2))

    cumulative = gram.cumsum(axis=0).cumsum(axis=1)
    total_cost = np.trace(gram) - gram.sum() / n

    for t in range(2, n - 1):
        left_sum = cumulative[t - 1, t - 1]
        right_sum = gram.sum() - cumulative[t - 1, -1] - cumulative[-1, t - 1] + left_sum

        left_cost = np.trace(gram[:t, :t]) - left_sum / t
        right_cost = np.trace(gram[t:, t:]) - right_sum / (n - t)
        out[t] = total_cost - left_cost - right_cost

    return out


def _median_heuristic(time_series: NDArray) -> float:
    """Median of the non-zero pairwise Euclidean distances."""
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    distances = pairwise_distances(x)
    return np.median(distances) 