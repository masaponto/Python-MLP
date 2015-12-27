#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation

from more_itertools import chunked


class MLP(BaseEstimator):

    def __init__(self,
                 hid_nums=[100],
                 epochs=1000,
                 r=0.5,
                 batch_size=200):
        self.hid_nums = hid_nums
        self.epochs = epochs
        self.r = r
        self.batch_size = batch_size

    def _sigmoid(self, x, a=1):
        """
        sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return 1 / (1 + np.exp(- a * x))

    def _dsigmoid(self, x, a=1):
        """
        diff sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return a * self._sigmoid(x) * (1.0 - self._sigmoid(x))

    def _ltov(self, n):
        """
        trasform label scalar to vector

        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        """
        def inltov(label):
            return [-1 if i != label else 1 for i in range(1, n + 1)]
        return inltov

    def _vtol(self, vec):
        """
        tranceform vector (list) to label
        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result
        """
        fix_num = lambda x: 1 if 1 == round(x, 0) else -1
        if self.out_num == 1:
            return fix_num(vec[0])
        else:
            v = list(vec)
            return int(v.index(max(v))) + 1

    def _add_bias(self, x_vs):
        """
        add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec
        """

        return np.c_[x_vs, np.ones(len(x_vs))]

    def _get_delta(self, w, delta, u):
        return self._dsigmoid(u) * np.dot(w.T, delta)

    def _predict(self, x):
        _x = x.T
        for w in self.ws:
            _x = self._sigmoid(np.dot(w, _x))
        return _x

    def predict(self, x):
        x = self._add_bias(x)
        y = x.T
        for w in self.ws:
            y = self._sigmoid(np.dot(w, y))
        y = y.T

        if self.out_num != 1:
            y = np.array(list(map(self._vtol, y)))

        return y

    def fit(self, X, y):
        self.data_num = len(X)
        assert(self.data_num > self.batch_size)

        self.out_num = max(y)

        # fix data
        if self.out_num != 1:
            y = np.array(list(map(self._ltov(self.out_num), y)))
        X = self._add_bias(X)

        # init hidden weight vecotors
        n = X.shape[1]
        self.ws = []
        for m in self.hid_nums:
            np.random.seed()
            w = np.random.uniform(-1., 1., (m, n))
            self.ws.append(w)
            n = m

        # init output weight vecotors
        np.random.seed()
        w = np.random.uniform(-1., 1., (self.out_num, n))
        self.ws.append(w)

        # fitting
        for n in range(self.epochs):
            for index in range(0, X.shape[0], self.batch_size):

                _x = X[index:index + self.batch_size]
                _y = y[index:index + self.batch_size]

                zs = []
                z = _x.T
                zs.append(z)
                us = []
                for w in self.ws[:len(self.ws) - 1]:
                    u = np.dot(w, z)
                    us.append(u)
                    z = self._sigmoid(u)
                    zs.append(z)

                deltas = []
                delta = _y.T - self._predict(_x)
                deltas.append(delta)
                for u, w in zip(reversed(us), reversed(self.ws)):
                    delta = self._get_delta(w, delta, u)
                    deltas.append(delta)

                dws = []
                for delta, z in zip(deltas, reversed(zs)):
                    dw = np.dot(delta, z.T) / self.batch_size
                    dws.append(dw)

                for i, dw in enumerate(reversed(dws)):
                    self.ws[i] = self.ws[i] - self.r * dw

def main():
    db_name = 'iris'

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    mlp = MLP(hid_nums=[10], epochs=10, batch_size=30)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    mlp.fit(X_train, y_train)
    re = mlp.predict(X_test)

    [print(w) for w in mlp.ws]
    print(re)

    score = sum([r == y for r, y in zip(re, data_set.target)]) / len(data_set.target)
    #print("general Accuracy %0.3f " % score)
    #print(score)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
