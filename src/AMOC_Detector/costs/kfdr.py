'''
A very old implementation of Kernel Fisher Discriminant ratio. 

This is done using the rbf kernel. 

This cost function is to be interacted with in the same way as all of the 
others in the repp 
'''
import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel  
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def kfdr(time_series, jitter = 0.025):
    """
    an implementation of kernel fisher discriminant ratio. 

    first, it fits the length scale using the median heurisitc 
    """
    length_scale = _median_heuristic(time_series)
    test_statistic = [0.0] * 10
    n = time_series.shape[0]
    K = rbf_kernel(time_series, gamma =1 / (2 * length_scale**2)) 
    
    # we can now loop 
    for n1 in tqdm(range(10, n-10)):
        # making the things we will be using 
        N_n, m_n = construct_N_n(n1, n),  construct_m_n(n1, n)
        NKN = N_n.T @ K @ N_n
        MKM = m_n.T @ K @ m_n 
        inversion = np.linalg.solve(jitter * np.eye(n) + NKN,N_n) 
        trace_mat = 1 / (n * jitter) * inversion @ K @ N_n.T

        # getting the things for our stat 
        stat = (kfdr_computation(inversion, MKM, jitter, K, N_n, m_n, n, n1) - d1(trace_mat)) / (np.sqrt(2) * d2(trace_mat))
        test_statistic.append(stat)
    return test_statistic +  [0.0] * 10

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