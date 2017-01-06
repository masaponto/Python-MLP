#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle


class TLMLP(BaseEstimator, ClassifierMixin):
    """
    Multi Layer Perceptron (Single hidden layer)

    """

    def __init__(self,
                 hid_num=10,
                 epochs=1000,
                 r=0.5):
        self.hid_num = hid_num
        self.epochs = epochs
        self.r = r

    def __sigmoid(self, x, a=1):
        return 1 / (1 + np.exp(- a * x))

    def __dsigmoid(self, x, a=1):
        return a * x * (1.0 - x)

    def __add_bias(self, X):
        return np.c_[X, np.ones(X.shape[0])]

    def __ltov(self, n, label):
        return [0 if i != label else 1 for i in range(1, n + 1)]

    def __calc_out(self, w, x):
        return self.__sigmoid(np.dot(w, x))

    def __out_error(self, z, y):
        return (z - y) * self.__dsigmoid(z)

    def __hid_error(self, z, eo):
        return np.dot(self.wo.T, eo) * self.__dsigmoid(z)

    def __w_update(self, w, e, z):
        e = np.atleast_2d(e)
        z = np.atleast_2d(z)
        return w - self.r * np.dot(e.T, z)

    def fit(self, X, y):
        """
        学習
        Args:
        X [[float]] array : featur vector
        y [int] array : class labels
        """

        self.out_num = max(y)

        y = np.array([self.__ltov(self.out_num, _y) for _y in y])

        X = self.__add_bias(X)

        np.random.seed()
        self.wh = np.random.uniform(-1., 1., (self.hid_num, X.shape[1]))
        self.wo = np.random.uniform(-1., 1., (self.out_num, self.hid_num + 1))

        for n in range(self.epochs):
            X, y = shuffle(X, y, random_state=np.random.RandomState())
            for _x, _y in zip(X, y):

                # forward phase
                zh = self.__calc_out(self.wh, _x)
                zh = np.insert(zh, zh.shape, 1)

                zo = self.__calc_out(self.wo, zh)

                # backward phase
                eo = self.__out_error(zo, _y)
                eh = self.__hid_error(zh, eo)

                eh = np.delete(eh, eh.shape[0] - 1)

                # weight update
                self.wo = self.__w_update(self.wo, eo, zh)
                self.wh = self.__w_update(self.wh, eh, _x)

        return self

    def predict(self, X):
        """
        Args:
        x_vs [[float]] array
        """
        X = self.__add_bias(X)
        z = self.__calc_out(self.wh, X.T)
        z = np.insert(z, z.shape[0], 1, axis=0)
        y = self.__calc_out(self.wo, z)

        if self.out_num == 1:
            y = y.T
            y[y < 0.5] = -1
            y[y >= 0.5] = 1
            return y
        else:
            return np.argmax(y.T, 1) + np.ones(y.T.shape[0])


def main():
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split

    db_name = 'iris'

    data_set = fetch_mldata(db_name)
    data_set.data = normalize(data_set.data)

    mlp = TLMLP(hid_num=5, epochs=10)

    X_train, X_test, y_train, y_test = train_test_split(
        data_set.data, data_set.target, test_size=0.4)

    mlp.fit(X_train, y_train)

    print("Accuracy %0.3f " % mlp.score(X_test, y_test))


if __name__ == "__main__":
    main()
