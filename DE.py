from __future__ import division

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd
import time

def density(x):
    """
    输入: 一个list, 是一系列一维样本
    输出: 基于DP的density估计
    """
    values = x
    values = np.array(values)
    values = (values - values.mean()) / values.std()
    
    N = len(values)
    K = 30
    SEED = int(time.time())

    def stick_breaking(beta):
        portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    with pm.Model() as model:
        alpha = pm.Gamma('alpha', 1., 1.)
        beta = pm.Beta('beta', 1., alpha, shape=K)
        w = pm.Deterministic('w', stick_breaking(beta))

        tau = pm.Gamma('tau', 1., 1., shape=K)
        lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
        mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
        obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                            observed=values)

    with model:
        trace = pm.sample(1000, random_seed=SEED, init='advi')

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_w = np.arange(K) + 1
    ax.bar(plot_w - 0.5, trace['w'].mean(axis=0), width=1., lw=0);
    ax.set_xlim(0.5, K);
    ax.set_xlabel('Component');
    ax.set_ylabel('Posterior expected mixture weight');

    post_pdf_contribs = sp.stats.norm.pdf(np.atleast_3d(x_plot),
                                        trace['mu'][:, np.newaxis, :],
                                        1. / np.sqrt(trace['lambda'] * trace['tau'])[:, np.newaxis, :])
    post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)
    return post_pdfs

if __name__ == "__main__":
    x = pd.read_csv(pm.get_data('old_faithful.csv'))
    x = x.iloc[:,1].values
    post_pdfs = density(x)