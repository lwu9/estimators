import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as gbr
from estimators.bandits import base
from typing import Optional

class Estimator(base.Estimator):
    examples_count: float
    weighted_reward: float

    def __init__(self, x_os, a_os, r_os, p_preds):
        # a_num: the size of the action space
        a_num = 
        # p_preds: an array of p_pred with length equal to a_num in a stationary fixed prediction policy
        m, p = x_os.shape
        self.examples_count = m
        self.x_os = x_os
        self.a_os = a_os
        self.r_os = r_os
        self.x_rct = [None] * p
        self.a_rct = []
        self.r_rct = []
        
        # direct method for the policy value estimation with only the OS data
        x_test_os = np.column_stack((self.x_os, self.x_os*0))
        reg = gbr().fit(self.x_os[self.a_os == 0, :], self.r_os[self.a_os == 0])
        r_est_os = reg.predict(x_test_os) * p_preds[0]
        for ai in range(1, a_num):
            reg = gbr().fit(self.x_os[self.a_os == ai, :], self.r_os[self.a_os == ai])
            r_est_os += reg.predict(x_test_os) * p_preds[ai]
        self.dm_reward = np.mean(r_est_all)
    
    def add_example(self, x: float, a: int, r: float, p_pred: float, count: float = 1.0):
        # p_pred: a number equal to P(A=a|X) in a stationary predition policy. (Future: an array of prediction probabilities with length equal to the action space for the changing prediction policy)
        self.x_rct = np.row_stack((self.x_rct, x))
        self.a_rct = np.append(self.a_rct, a)
        self.r_rct = np.append(self.r_rct, r)
        self.x_test = np.append(self.x, self.x * 0)
        self.examples_count += count
        x_int = np.row_stack((np.column_stack((self.x_rct[1:, :], self.x_rct[1:, :] * 0)), np.column_stack((self.x_os, self.x_os))))
        # Gradient boosting regression used in the direct method
        reg_a = gbr().fit(x_int[np.append(self.a_rct, self.a_os) == a, :], np.append(self.r_rct[self.a_rct == a], self.r_os[self.a_os == a]))
        r_est_a = reg_a.predict(self.x_test)
        self.dm_reward += r_est_a * p_pred * count

    def get(self):
        return self.dm_reward/self.example_count