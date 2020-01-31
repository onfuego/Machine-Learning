#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, mean_squared_error, confusion_matrix, accuracy_score, recall_score, f1_score, precision_recall_curve

color_palette_divergent = LinearSegmentedColormap.from_list('ee', ['#E27872', '#F9F9F8', '#509A9A'])
color_palette_discrete = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
color_palette_sequential = [ '#ece3f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016c59', '#014636']
markers = ['o', '^', '*','H', 'P', 'D', 'X', 'h', 'p', 'd', 'c']

class ClassificationModelMetrics(object):
    """docstring for ClassificationModelMetrics"""

    def __get_mu_sigma(self, train_vector, test_vector):
        """TODO: Docstring for __get_mu_sigma.

        :test_vector: TODO
        :train_vector: TODO
        :returns: TODO

        """
        tmp_mu_train = np.mean(train_vector, axis=1)
        tmp_sigma_train = np.std(train_vector, axis=1)
        tmp_mu_test = np.mean(test_vector, axis=1)
        tmp_sigma_test = np.std(test_vector, axis=1)
        return tmp_mu_train, tmp_sigma_train, tmp_mu_test, tmp_sigma_test

    def __generate_mesh_grid(self, mat, x1, x2):
        """TODO: Docstring for __generate_mesh_grid.

        :mat: TODO
        :x1ODO
        :x2: TODO
        :returns: TODO

        """
        tmp_X = mat.loc[:, [x1, x2]]

        x_0, x_1 = np.meshgrid(
            np.linspace(np.min(tmp_X[x1]), np.max(tmp_X[x1]), num=100).reshape(-1, 1),
            np.linspace(np.min(tmp_X[x2]), np.max(tmp_X[x2]), num=100).reshape(-1, 1)
        )

        return x_0, x_1

    def __hyperparams_dictionary(self, model_name):
        """TODO: Docstring for __hyperparams_dictionary.

        :model_name: TODO
        :returns: TODO

        """
        tmp = {
            'LogisticRegression': ['C', np.linspace(0.1, 1000.0, 50, dtype=float), "linear"],
            'SVC': ['gamma', np.logspace(-6, -1, 50, dtype=float), "semilog"],
            'DecisionTreeClassifier':['max_depth', np.linspace(1, 100, 50, dtype=int), "linear"],
            'SVM': ['C', np.linspace(0.1, 1000.0, 50, dtype=float), "linear"],
            'LinearSVC':['C', np.linspace(0.1, 1000.0, 50, dtype=float), "linear"],
            'NuSVC':['nu', np.linspace(0, 1, 50, dtype=float), "linear"],
            'MLPClassifier': ['alpha', np.linspace(0.0001, 1, 50, dtype=float), "linear"],
            'BernoulliNB': ['alpha', np.linspace(0, 1, 50, dtype=float), "linear"],
            'MultinomialNB': ['alpha', np.linspace(0, 1, 50, dtype=float), "linear"],
            'LinearDiscriminantAnalysis': ['shrinkage', np.linspace(0, 1, 50, dtype=float), "linear"],
            'QuadraticDiscriminantAnalysis': ['reg_param', np.linspace(0, 1, 50, dtype=float), "linear"],
            'SGDClassifier': ['alpha', np.linspace(0.0001, 1, 50, dtype=float), "linear"],
            'RidgeClassifier': ['alpha', np.linspace(0.0001, 1, 50, dtype=float), "linear"]
        }
        return tmp[model_name]

    def __init__(self, model, X_mat, y_vec, test_size_percentage=.33, random_seed=11238):
        self.model = model
        self.X_mat = X_mat
        self.y_vec = y_vec
        self.test_size = test_size_percentage
        self.random_seed = random_seed
        self.X_mat_train, self.X_mat_test,self.y_vec_train, self.y_vec_test = train_test_split(self.X_mat,
                                                                                               self.y_vec,
                                                                                               test_size=self.test_size,
                                                                                               random_state=self.random_seed, shuffle=True)

    def plot_learning_curve(self, cv=None, train_n=np.linspace(0.1, 1.0, 5, dtype=float), score=None):
        """TODO: Docstring for plot_learning_curve.

        :cv: TODO
        :train_n: TODO
        :1.0: TODO
        :5: TODO
        :dtype: TODO
        :score: TODO
        :signal_error: TODO
        :returns: TODO

        """
        n_size, train_vector, test_vector = learning_curve(self.model,
                                                           self.X_mat,
                                                           self.y_vec,
                                                           cv=cv,
                                                           train_sizes=train_n,
                                                           scoring=score)

        mu_train_vector, sigma_train_vector, mu_test_vector, sigma_test_vector = self.__get_mu_sigma(train_vector, test_vector)

        plt.plot(n_size, mu_train_vector, color=color_palette_discrete[2],
                 linestyle=":", lw=3, label='Training')

        if cv is not None:
            plt.plot(n_size, mu_test_vector, color=color_palette_discrete[1],
                     linestyle=":", lw=3, label='Testing on {}-fold CV'.format(cv))
        else:
            plt.plot(n_size, mu_test_vector, color=color_palette_discrete[1],
                     linestyle=":", lw=3, label='Testing on 3-fold CV')


        if score is not None:
            plt.ylabel('Scoring method: {}'.format(score))
        else:
            plt.ylabel('Score')

        plt.xlabel('Training set size')
        plt.legend()

    def plot_validation_curve(self, cv=10, score='accuracy', kind=None, param_name=None, param_range=None):
        """TODO: Docstring for plot_validation_curve.

        :cv: TODO
        :score: TODO
        :param_name: TODO
        :param_range: TODO
        :returns: TODO

        """
        tmp_model_id = str(self.model).split('(')[0]
        if param_name is None and param_range is None and kind is None:
            fetch_params = self.__hyperparams_dictionary(tmp_model_id)
        else:
            fetch_params = [param_name, param_range, kind]

        train_vector, test_vector = validation_curve(self.model,
                                                     self.X_mat,
                                                     self.y_vec,
                                                     param_name=fetch_params[0],
                                                     param_range=fetch_params[1],
                                                     cv=cv,
                                                     scoring=score)

        mu_train_vector, sigma_train_vector, mu_test_vector, sigma_test_vector = self.__get_mu_sigma(train_vector, test_vector)

        if 'linear' in fetch_params[2]:
            plt.plot(fetch_params[1], mu_train_vector, color=color_palette_discrete[2],
                     linestyle=':', lw=3, label="Training")
            plt.plot(fetch_params[1], mu_test_vector, color=color_palette_discrete[1],
                     linestyle=':', lw=3, label='Testing on {}-fold CV'.format(cv))
        elif 'semilog' in fetch_params[2]:
            plt.semilogx(fetch_params[1], mu_train_vector, color=color_palette_discrete[2],
                         linestyle=':', lw=3, label="Training")
            plt.semilogx(fetch_params[1], mu_test_vector, color=color_palette_discrete[1],
                         linestyle=':', lw=3, label='Testing on {}-fold CV'.format(cv))
        else:
            raise ValueError("Missing argument")

        if score is not None:
            plt.ylabel('Scoring method: {}'.format(score))
        else:
            plt.ylabel('Score')

        plt.xlabel("Hyperparameter: {}".format(fetch_params[0]))
        plt.legend()

    def pr_contour_plot(self, x1, x2, fill_contours = False):
        """TODO: Docstring for pr_contour_plot.

        :target: TODO
        :x1: TODO
        :x2: TODO
        :**model_kwargs: TODO
        :returns: TODO

        """
        tmp_X = self.X_mat.loc[:, [x1, x2]]
        tmp_model = self.model.fit(tmp_X, self.y_vec)
        x_0, x_1 = self.__generate_mesh_grid(tmp_X, x1, x2)
        custom_cmap = ListedColormap(color_palette_sequential)
        map_x = np.c_[x_0.ravel(), x_1.ravel()]
        predict_y_pr = tmp_model.predict_proba(map_x)
        predict_y = tmp_model.predict(map_x)
        boundaries_pr = predict_y_pr[:, 1].reshape(x_1.shape)
        boundaries_y = predict_y.reshape(x_0.shape)

        for i in self.y_vec.unique():
            plt.plot(tmp_X[self.y_vec == i][x1],
                     tmp_X[self.y_vec == i][x2],
                     '.',
                     marker=markers[i],
                     color=color_palette_discrete[i],
                     label="{}".format(i),
                     alpha=.8)

        if fill_contours is True:
            custom_cmap = LinearSegmentedColormap.from_list('lista', color_palette_sequential)
            plt.contourf(x_0, x_1, boundaries_pr, cmap=custom_cmap)
            plt.colorbar()
            plt.clim(0, 1)
        else:
            vis_boundaries = plt.contour(x_0, x_1, boundaries_pr, cmap = custom_cmap)
            plt.clabel(vis_boundaries, inline=1, lw=4)

        plt.legend(framealpha=0.5, edgecolor='slategrey', fancybox=True)
        plt.xlabel(x1)
        plt.ylabel(x2)

    def prec_rec_plot(self):
        """TODO: Docstring for prec_rec_plot.
        :returns: TODO

        """
        y_hat = self.model.fit(self.X_mat_train, self.y_vec_train).predict(self.X_mat_test)
        tmp_prec, tmp_recall, threshold = precision_recall_curve(self.y_vec_test, y_hat)
        plt.plot(tmp_recall, tmp_prec, color_palette_discrete[1], lw=3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')

    def roc_plot(self, dummy_classifier=False):
        """TODO: Docstring for roc_plot.

        :dummy_classifier: TODO
        :returns: TODO

        """
        y_hat = self.model.fit(self.X_mat_train, self.y_vec_train).predict(self.X_mat_test)
        false_positive, true_positive, threshold = roc_curve(self.y_vec_test, y_hat)
        tmp_roc_auc = round(roc_auc_score(self.y_vec_test, y_hat), 3)
        plt.plot(false_positive, true_positive, color_palette_discrete[1], lw=3, label="AUC: {}".format(tmp_roc_auc))
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.axis([0, 1, 0, 1])
        plt.legend(framealpha=0.5, edgecolor='slategrey', fancybox=True)

        if dummy_classifier is True:
            plt.plot([0, 1], [0, 1])

    def confusion_plot(self):
        """TODO: Docstring for confusion_plot.

        :f: TODO
        :returns: TODO

        """
        y_hat = self.model.fit(self.X_mat_train, self.y_vec_train).predict(self.X_mat_test)
        tmp_confused = confusion_matrix(self.y_vec_test, y_hat)
        custom_cmap = LinearSegmentedColormap.from_list('lista', color_palette_sequential)

        plt.matshow(tmp_confused, cmap=custom_cmap)
        plt.colorbar()

        for i in range(len(tmp_confused)):
            for j in range(len(tmp_confused)):
                test = plt.text(j, i, tmp_confused[i, j])
