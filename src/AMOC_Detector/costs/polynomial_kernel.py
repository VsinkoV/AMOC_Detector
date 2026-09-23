import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import polynomial_kernel as sklearn_polynomial_kernel


def polynomial_kernel(time_series: NDArray, length_scale: float | None = None,
                      degree: int = 3, coef0: float = 1.0) -> NDArray:
    """
    Kernel based reduction in cost from adding one changepoint.

    Vectorised in the same way as the RBF cost: prefix sums of the Gram
    matrix and of its diagonal replace the per-iteration slicing and
    gram.sum() calls, taking the curve from cubic to quadratic in n.
    Numerically identical to the previous version.
    """
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    n = len(x)
    out = np.zeros(n)
    if length_scale is None:
        length_scale = _median_heuristic(x)
    gram = sklearn_polynomial_kernel(x, gamma=1 / (2 * length_scale ** 2),
                                     degree=degree, coef0=coef0)

    cumulative = np.zeros((n + 1, n + 1))
    cumulative[1:, 1:] = gram.cumsum(axis=0).cumsum(axis=1)
    diag_cs = np.concatenate(([0.0], np.cumsum(np.diag(gram))))

    grand_sum = cumulative[n, n]
    trace = diag_cs[n]
    total_cost = trace - grand_sum / n

    t = np.arange(2, n - 1)
    if t.size:
        left_sum = cumulative[t, t]
        right_sum = grand_sum - cumulative[t, n] - cumulative[n, t] + left_sum
        left_cost = diag_cs[t] - left_sum / t
        right_cost = (trace - diag_cs[t]) - right_sum / (n - t)
        out[2:n - 1] = total_cost - left_cost - right_cost

    return out


def _median_heuristic(time_series: NDArray) -> float:
    """Median of the non-zero pairwise Euclidean distances."""
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    distances = pairwise_distances(x)
    return np.median(distances)
