#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec3_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Assorted functions for naive bayes - adl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB

def simulate_bayes(x_axis_n = 100, sample_size = 500, p = .7, prior_pr = .5, prior_size = 1000):
    """docstring for simulate_bayes"""
    xaxis = np.linspace(0, 1, x_axis_n)
    empirical_sample = np.random.binomial(n=1, p=p, size=sample_size)
    likelihood = np.array([np.product(stats.bernoulli.pmf(empirical_sample, i)) for i in xaxis])
    likelihood_point = np.mean(likelihood)
    prior_sample = np.random.binomial(n=1, p=prior_pr, size=prior_size)
    prior_prob = np.array([np.product(stats.bernoulli.pmf(prior_sample, i)) for i in xaxis])
    prior_prob = prior_prob / np.sum(prior_prob)
    posterior_prob = [likelihood[i] * prior_prob[i] for i in range(prior_prob.shape[0])]

    plt.subplot(3, 1, 1)
    plt.vlines(xaxis, 0, likelihood, color='tomato')
    plt.title('Verosimilitud')
    plt.subplot(3, 1, 2)
    plt.vlines(xaxis, 0, prior_prob, color='dodgerblue')
    plt.title("Probabilidad a priori")
    plt.subplot(3, 1, 3)
    plt.vlines(xaxis, 0, posterior_prob, color='violet')
    plt.title("Probabilidad a posteriori")
    # plt.suptitle('A posteriori ' +  r'$\propto$' + 'Verosimilitud' + r'$\times$' + 'A priori')
    plt.tight_layout()

def deaggregate_statistics(dataframe):
    """Given a frequency, multiply attributes combination row-wise and generate a new dataframe

    :dataframe: TODO
    :returns: TODO

    """
    final_gender = []
    final_dept = []
    final_admit = []

    for _, row_serie in dataframe.iterrows():
        for _ in range(1, row_serie[3] + 1):
            final_admit.append(row_serie[0])
            final_gender.append(row_serie[1])
            final_dept.append(row_serie[2])

    return pd.DataFrame({
        'Admit': final_admit,
        'Gender': final_gender,
        'Dept': final_dept
    })

def compare_priors(X_train, X_test, y_train, y_test, prior):
    """TODO: Docstring for compare_priors.

    :prior: TODO
    :returns: TODO

    """
    tmp_clf = BernoulliNB(class_prior=prior)
    tmp_clf.fit(X_train, y_train)
    tmp_class = tmp_clf.predict(X_test)
    tmp_pr = tmp_clf.predict_proba(X_test)[:, 1]
    tmp_acc = accuracy_score(y_test, tmp_class).round(3)
    tmp_rec = recall_score(y_test, tmp_class).round(3)
    tmp_prec = precision_score(y_test, tmp_class).round(3)
    tmp_f1 = f1_score(y_test, tmp_class).round(3)
    tmp_auc = roc_auc_score(y_test, tmp_pr).round(3)
    print("A priori: {0}\nAccuracy: {1}\nRecall: {2}\nPrecision: {3}\nF1: {4}\nAUC: {5}\n".format(prior, tmp_acc, tmp_rec, tmp_prec, tmp_f1, tmp_auc))

