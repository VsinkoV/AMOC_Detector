'''
A very old implementation of Kernel Fisher Discriminant ratio. 

This is done using the rbf kernel. 

This cost function is to be interacted with in the same way as all of the 
others in the repp 
'''
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances


def kfdr(time_series, ts , jitter = 0.025):
    """
    an implementation of kernel fisher discriminant ratio.

    """
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    n = x.shape[0]
    length_scale = _median_heuristic(x)
    K = rbf_kernel(x, gamma=1 / (2 * length_scale**2))
    return _kfdr_from_gram(K, ts, n, jitter)


def kfdr_curve(time_series, jitter=0.025, jump=1):
    """
    Kernel Fisher Discriminant Ratio scored across every candidate
    changepoint 
    """
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    n = x.shape[0]
    length_scale = _median_heuristic(x)
    K = rbf_kernel(x, gamma=1 / (2 * length_scale**2))

    out = np.zeros(n)
    for t in range(2, n - 1, jump):
        out[t] = _kfdr_from_gram(K, t, n, jitter)
    return out


def _kfdr_from_gram(K, t, n, jitter):
    """KFDR statistic for one candidate changepoint t, given a precomputed Gram matrix."""
    N_n, m_n = construct_N_n(t, n), construct_m_n(t, n)
    NKN = N_n.T @ K @ N_n
    MKM = m_n.T @ K @ m_n
    inversion = np.linalg.solve(jitter * np.eye(n) + NKN, N_n)
    trace_mat = 1 / (n * jitter) * inversion @ K @ N_n.T
    return (kfdr_computation(inversion, MKM, jitter, K, N_n, m_n, n, t) - d1(trace_mat)) / (np.sqrt(2) * d2(trace_mat))


# many many helper functions 
def construct_m_n(n1, n):
    n2 = n - n1
    m_n = np.zeros(n)
    m_n[:n1] = -1 / n1
    m_n[n1:] = 1 / n2
    return m_n

def construct_N_n(n1, n):
    n2 = n - n1
    P1 = P(n1)
    P2 = P(n2)
    N_n = np.block([
        [P1, np.zeros((n1, n2))],
        [np.zeros((n2, n1)), P2]
    ])
    return N_n

def P(l):
    I = np.eye(l)
    ones = np.ones((l, 1))
    return I - (1 / l) * (ones @ ones.T)

def kfdr_computation(inversion, MKM, jitter, K, N_n, m_n, n, n1):
    return 1/jitter * (MKM - 1/n * m_n.T @ K @ N_n @ inversion @ K @ m_n) * (n1 * (n-n1)) / n 

def d1(trace_mat):
    return np.trace(trace_mat)

def d2(trace_mat):
    return np.trace(trace_mat @ trace_mat.T)

def _median_heuristic(time_series) -> float:
    """Median of pairwise Euclidean distances."""
    #Ensures there are no isses in the median
    x = np.asarray(time_series, dtype=float).reshape(len(time_series), -1)
    distances = pairwise_distances(x)
    return np.median(distances)

