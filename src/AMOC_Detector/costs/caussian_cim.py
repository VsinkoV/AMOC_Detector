"""
Take some time series, which is assumed to be gaussian, and see what the 
improvement in cost is if you allow it to add in a changepoint. 

Global constants such as variance will be ignored. 
"""
import numpy as np 
from numpy.typing import NDArray

def gaussian_mean_change(time_series: NDArray) -> NDArray:
    """Cost reduction from adding one Gaussian mean change."""
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    n = len(x)
    out = np.zeros(n)
    cs = np.vstack((np.zeros(x.shape[1]), np.cumsum(x, axis=0)))

    for t in range(2, n - 1):
        left, right = cs[t] / t, (cs[n] - cs[t]) / (n - t)
        out[t] = t * (n - t) / n * np.sum((left - right) ** 2)

    return out

def gaussian_variance_change(time_series: NDArray) -> NDArray:
    x = np.asarray(time_series,dtype= float).reshape(len(time_series),-1)
    n = len(x)
    out = np.zeros(n)
    cs = np.vstack((np.zeros(x.shape[1]), np.cumsum(x**2, axis = 0)))

    for t in range(2,n-1):
        stat = n* np.log(cs[n]/n)- t* np.log(cs[t]/t) - (n-t)*np.log((cs[n]-cs[t])/(n-t))
        out[t] = np.sum(stat)

    return out

def gaussian_meanvar_change(time_series: NDArray) -> NDArray:
    """Cost reduction from adding one changepoint where mean and variance are
    both allowed to change - joint Gaussian log-likelihood ratio, with each
    segment's variance centered on its own mean (unlike gaussian_variance_change,
    which assumes a fixed/zero mean throughout)."""
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    n = len(x)
    out = np.zeros(n)
    cs1 = np.vstack((np.zeros(x.shape[1]), np.cumsum(x, axis=0)))
    cs2 = np.vstack((np.zeros(x.shape[1]), np.cumsum(x**2, axis=0)))

    def seg_var(a, b):
        m = b - a
        s1 = cs1[b] - cs1[a]
        s2 = cs2[b] - cs2[a]
        return s2 / m - (s1 / m) ** 2

    total_var = seg_var(0, n)
    for t in range(2, n - 1):
        left_var = seg_var(0, t)
        right_var = seg_var(t, n)
        stat = n * np.log(total_var) - t * np.log(left_var) - (n - t) * np.log(right_var)
        out[t] = np.sum(stat)

    return out
