import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as gbr
import numpy.random as rand

from estimators.bandits import base
from estimators.bandits import integrative
from typing import Optional

def simulate(N, p, p0, n, m, b):
    # Generate the population data
    diag = np.ones(p)
    cov = np.diag(diag)
    x = rand.multivariate_normal(diag, cov, (N, ))
    u = rand.standard_normal(N)
    mu_x = x[:, 0] + u
    coef = np.append(np.ones(p - p0), np.zeros(p0))
    tau = np.matmul(x, coef) + np.matmul(np.power(x, 2), coef)
    y1 = mu_x + tau + rand.standard_normal(N)
    y0 = mu_x + rand.standard_normal(N)
    # Evaluation policy \pi(a|X) = 0.5, a = 0, 1
    v_pi = np.mean(y1+y0) * 0.5

    # Generate the RCT
    #exp_term = np.exp(-e0 - x[:, 0] + x[:, 1])
    #s_prob = exp_term / (1 + exp_term)
    #s1 = rand.binomial(1, s_prob, N)
    #s1_idx = s1 == 1
    #n = sum(s1)
    s1_idx = rand.choice(N, n, replace=False)
    x_rct = x[s1_idx, :]
    exp_term2 = np.exp(-1 - x_rct[:, 0] + x_rct[:, 1])
    a_rct_prob = exp_term2 / (1 + exp_term2)
    a_rct = rand.binomial(1, a_rct_prob, n)
    y_rct = y1[s1_idx] * a_rct + y0[s1_idx] * (1 - a_rct)
    # IPS value estimate with only RCT
    v_pi_ips_rct = np.mean(0.5*y_rct/(a_rct*a_rct_prob + (1-a_rct)*(1-a_rct_prob)))
    # SNIPS value estimate with only RCT
    v_pi_snips_rct = np.sum(0.5*y_rct/(a_rct*a_rct_prob + (1-a_rct)*(1-a_rct_prob)))/sum(0.5/(a_rct*a_rct_prob + (1-a_rct)*(1-a_rct_prob)))

    # Generate the OS
    s0_idx = rand.choice(N, m, replace=False)
    x_os = x[s0_idx, :]
    exp_term3 = np.exp(-1 - x_os[:, 0] + x_os[:, 1] + b * u[s0_idx])
    a_os_prob = exp_term3 / (1 + exp_term3)
    a_os = rand.binomial(1, a_os_prob, m)
    y_os = y1[s0_idx] * a_os + y0[s0_idx] * (1 - a_os)

    return {'x_test': x, 'tau': tau, 'x_rct': x_rct, 'a_rct': a_rct, 'y_rct': y_rct, 'a_rct_prob': a_rct_prob,
            'x_os': x_os, 'a_os': a_os, 'y_os': y_os, 'a_os_prob': a_rct_prob, 'y1': y1, 'y0': y0, 
            'v_pi': v_pi, 'v_pi_ips_rct': v_pi_ips_rct, 'v_pi_snips_rct': v_pi_snips_rct}


m = 2500
n = 500
a_num = 2
data = simulate(N=100000, p=9, p0=1, n=n, m=m, b=2.5)
print(data['x_test'].shape)
print(data['v_pi'], data['v_pi_snips_rct'])
p_preds = np.ones((m, a_num))/a_num
est = integrative.Estimator(data['x_os'], data['a_os'], data['y_os'], p_preds)
for i in range(n):
    a = data['a_rct'][i]
    p_pred_arr = np.array([0.5, 0.5])
    est.add_example(data['x_rct'][i, :], a, data['y_rct'][i], p_pred_arr)
    est.dm_int_each(a, p_pred_arr)
    if (i+1)%50 == 0:
        print(i, est.dm_int_arr(), est.get())
print(est.os(), est.rct())
