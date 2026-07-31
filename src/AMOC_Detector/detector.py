"""
Module for running a single iteration of binary segmentation 

most of the magic is happening in the cost functions. 
"""

import numpy as np 

class Detector:
    def __init__(
            self,
            cost_function, 
            penalty, 
    ):
        self.cost_function = cost_function
        self.penalty = penalty 

    def detect(self, time_series):

        improvements = self.cost_function(time_series)

        best_improvement = np.max(improvements)
        if best_improvement >= self.penalty:
            return np.where(improvements == best_improvement)