#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: lec2_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary functions for GAM and regularization
"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings(action='ignore')

def polynomial_degrees(m=50):
    """TODO: Docstring for polynomial_degrees.
    :returns: TODO

    """
    scatter_kws = {'color':'slategrey'}
    line_kws= {'color': 'tomato', 'linewidth': 3}
    np.random.seed(11238)
    X_mat = 3 * np.random.rand(m, 1) / 3.0
    y = 1 - 3 * X_mat + np.random.randn(m, 1) / 1

    plt.subplot(2, 3, 1)
    sns.regplot(X_mat[:, 0], y[:, 0], order=1, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    y_lim = plt.ylim()
    plt.title(r'$y = \beta_{0} + \beta_{1}X_{1} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 2)
    sns.regplot(X_mat[:, 0], y[:, 0], order=3, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{3} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)


    plt.subplot(2, 3, 3)
    sns.regplot(X_mat[:, 0], y[:, 0], order=5, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{5} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 4)
    sns.regplot(X_mat[:, 0], y[:, 0], order=7, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{7} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 5)
    sns.regplot(X_mat[:, 0], y[:, 0], order=10, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{10} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)

    plt.subplot(2, 3, 6)
    sns.regplot(X_mat[:, 0], y[:, 0], order=20, scatter_kws=scatter_kws, line_kws=line_kws)
    sns.despine()
    plt.ylim(y_lim)
    plt.title(r'$y = \beta_{0} +\sum_{j=1}^{20} \beta_{j} X_{i}^{j} + \varepsilon_{i}$', y=1.1)
    plt.tight_layout()
