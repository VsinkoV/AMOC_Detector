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
        left,right = cs[t] , (cs[n]-cs[t])
        out[t] = n* np.log(cs[n]/n)- t* np.log(cs[t]/t) - (n-t)*np.log((cs[n]-cs[t])/(n-t))


    