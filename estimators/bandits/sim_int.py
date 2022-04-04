import numpy as np
import numpy.random as rand

from estimators.bandits import integrative


def simulate(N, p, p0, n, m, b):
    # Generate the population data
    diag = np.ones(p)
    cov = np.diag(diag)
    x = rand.multivariate_normal(diag, cov, (N, ))
    u = rand.normal(0, 1, N)
    mu_x = x[:, 0] + u
    coef = np.append(np.ones(p - p0), np.zeros(p0))
    tau = np.matmul(x, coef) + np.matmul(np.power(x, 2), coef)
    y1 = mu_x + tau + rand.standard_normal(N)
    y0 = mu_x + rand.standard_normal(N)
    # Evaluation policy \pi(a|X) = 0.5, a = 0, 1
    v_pi = np.mean(y1+y0) * 0.5

    # Generate the RCT
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

import random
m = 2500
n = 10000
a_num = 2
p_preds = np.ones((m, a_num))/a_num
Rep = 100
ns = [500, 1000, 2500, 5000, 10000]
result = np.zeros((Rep, len(ns)*2 + 1))
for seed in range(Rep):
    random.seed(seed)
    data = simulate(N=100000, p=9, p0=1, n=n, m=m, b=2.5)
    est = integrative.Estimator(data['x_os'], data['a_os'], data['y_os'], p_preds)
    v_os = est.os()
    result[seed, 0] = abs(v_os - data['v_pi'])
    #print(v_os, data['v_pi'], data['v_pi_snips_rct'])
    ress = []
    for i in range(n):
        a = data['a_rct'][i]
        p_pred_arr = np.array([0.5, 0.5])
        est.add_example(data['x_rct'][i, :], a, data['y_rct'][i], p_pred_arr)
        if i+1 in ns:
            res = np.append(abs(data['v_pi'] - est.get()), 
                            abs(data['v_pi'] - est.rct()))
            ress = np.append(ress, res)
    result[seed, 1:] = ress
    if seed%10 == 0:
        print(seed, result[seed, :])
np.mean(result, 0)
# visualize the results
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
yerr = np.std(result, 0) / np.sqrt(Rep) * 1.96
means = np.mean(result, 0)
plt.errorbar(ns, means[np.array([1,3,5,7,9])], 
            yerr = yerr[np.array([1,3,5,7,9])], label='dm_gbr_int')
plt.errorbar(ns, means[np.array([2,4,6,8,10])], 
            yerr = yerr[np.array([2,4,6,8,10])], label='dm_gbr_rct')
plt.errorbar(ns, np.ones((5))*means[0], 
            yerr = np.ones((5))*yerr[0], label='dm_gbr_os')

plt.ylabel('Absolute error of value estimate')
plt.xlabel('Sample size of the experimental data')
plt.title('m = 2500')
plt.legend()
plt.show()    