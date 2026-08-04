import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import laplacian_kernel as sklearn_laplacian_kernel
from sklearn.metrics.pairwise import manhattan_distances

def Laplacian_kernel(time_series: NDArray, length_scale: float | None = None) -> NDArray:
    #Turn it from (n,) to (n,1)
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    n = len(x)
    out = np.zeros(n)
    if length_scale is None:
        length_scale = _median_heuristic(x)
    gram = sklearn_laplacian_kernel(x, gamma=1 / length_scale)
        #Here we sum together all the values in our gram matrix for simplicities sake
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
    #Ensures there are no isses in the median
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    distances = pairwise_distances(x)
    return np.median(distances) 

