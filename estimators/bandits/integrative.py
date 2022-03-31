import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as gbr
from estimators.bandits import base
from typing import Optional

import pdb

class Estimator(base.Estimator):
    examples_count: float
    weighted_reward: float

    def __init__(self, x_os, a_os, r_os, p_preds):
        # a_num: the size of the action space
        # p_preds: an 2D array with size m * a_num storing P(A=a|X=x), for all a in the action space and all x in the OS, in a stationary fixed prediction policy
        self.a_num = len(p_preds[0, :])
        self.p_preds = p_preds
        self.m, p = x_os.shape
        self.examples_count = self.m
        self.dm_reward = 0
        self.x_os = x_os
        self.a_os = a_os
        self.r_os = r_os
        self.x_rct = [None] * p
        self.a_rct = []
        self.r_rct = []
        self.regs = [None] * self.a_num
        self.x_int = np.column_stack((self.x_os, self.x_os))
        self.x_int_test = np.column_stack((self.x_os, self.x_os * 0))
        self.x_test = None
        
    
    def dm(self, X, A, R, a_num, X_test, p_preds):
        # direct method for the policy value estimation 
        r_est = 0
        for ai in range(a_num):
            reg = gbr().fit(X[A==ai, :], R[A==ai])
            r_est += np.dot(reg.predict(X_test), p_preds[:, ai])
        return r_est/len(A)

    def rct(self):
        return self.dm(self.x_rct[1:, :], self.a_rct, self.r_rct, self.a_num, self.x_rct[1:, :], self.p_preds[self.m:, :])
    
    def os(self):
        return self.dm(self.x_os, self.a_os, self.r_os, self.a_num, self.x_os, self.p_preds[:self.m, :])

    def add_example(self, x: float, a: int, r: float, p_pred_arr: float, count: float = 1.0):
        # p_pred_arr: an array with length a_num, represents P(A=a|X), for all a in the action space, in a stationary predition policy. (Future: an array of prediction probabilities with length equal to the action space for the changing prediction policy)
        self.x_rct = np.row_stack((self.x_rct, x))
        self.a_rct = np.append(self.a_rct, a)
        self.r_rct = np.append(self.r_rct, r)
        self.p_preds = np.row_stack((self.p_preds, p_pred_arr))
        self.x_test = np.append(x, x * 0).reshape(1, -1)
        self.examples_count += count
        self.x_int = np.row_stack((self.x_int, self.x_test))
        self.x_int_test = np.row_stack((self.x_int_test, self.x_test))
    
    def dm_int_arr(self):
        return self.dm(self.x_int, np.append(self.a_os, self.a_rct), np.append(self.r_os, self.r_rct), 
                       self.a_num, self.x_int_test, self.p_preds)

    def dm_int_each(self, a, p_pred_arr):
        # update the fitted reward function for the each coming example
        if len(self.a_rct) == 1:
            for ai in range(self.a_num):
                reg = gbr().fit(self.x_int[np.append(self.a_os, self.a_rct) == ai, :], 
                                np.append(self.r_os[self.a_os == ai], self.r_rct[self.a_rct == ai]))
                self.dm_reward += np.dot(reg.predict(self.x_int_test), self.p_preds[:, ai])
                self.regs[ai] = reg
        else:
            self.regs[a] = gbr().fit(self.x_int[np.append(self.a_os, self.a_rct) == a, :],
                                     np.append(self.r_os[self.a_os == a], self.r_rct[self.a_rct == a]))
            for ai in range(self.a_num):    
                self.dm_reward += self.regs[ai].predict(self.x_test) * p_pred_arr[ai]
        
    def get(self):
        return self.dm_reward/self.examples_count