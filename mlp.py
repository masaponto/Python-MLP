#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation


class MLP(BaseEstimator):

    def __init__(self,
                 hid_nums=[100],
                 epochs=1000,
                 r=0.5,
                 batch_size=20):

        self.hid_nums = hid_nums
        self.epochs = epochs
        self.r = r
        self.batch_size = batch_size
        self.ws = []

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
        derivative of sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return a * self._sigmoid(x) * (1.0 - self._sigmoid(x))

    def _ltov(self, n):
        """
        trasform label(integer) to vector (list)

        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label

        Examples:
        >>> mlp = MLP(10, 3)
        >>> mlp._ltov(3)(1)
        [1, 0, 0]
        >>> mlp._ltov(3)(2)
        [0, 1, 0]
        >>> mlp._ltov(3)(3)
        [0, 0, 1]
        """
        def inltov(label):
            if n == 1:
                return 1 if label == 1 else 0
            else:
                return [0 if i != label else 1 for i in range(1, n + 1)]
        return inltov

    def _vtol(self, vec):
        """
        tranceform vector(numpy array) to label (integer)
        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Examples
        >>> mlp = MLP(10, 3)
        >>> mlp.out_num = 3
        >>> mlp._vtol([1, -1, -1])
        1
        >>> mlp._vtol([-1, 1, -1])
        2
        >>> mlp._vtol([-1, -1, 1])
        3
        """
        if self.out_num == 1:
            return 1 if 1 == np.around(vec) else 0

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

        Examples:
        >>> mlp = MLP([10])
        >>> mlp._add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[ 1.,  2.,  3.,  1.],
               [ 1.,  2.,  3.,  1.]])
        """
        return np.c_[x_vs, np.ones(len(x_vs))]

    def _get_delta(self, w, delta, u):
        return self._dsigmoid(u) * np.dot(w.T, delta)

    def _predict(self, x):
        y = x.T
        for w in self.ws:
            y = self._sigmoid(np.dot(w, y))
        return y

    def predict(self, x):
        x = self._add_bias(x)
        y = x.T
        for w in self.ws:
            y = self._sigmoid(np.dot(w, y))

        return np.array(list(map(self._vtol, y.T)))

    def fit(self, X, y):
        # number of training data
        self.data_num = X.shape[0]
        assert(self.data_num > self.batch_size)

        # numer of output neurons
        self.out_num = max(y)

        # fix data label(integer) to array
        y = np.array(list(map(self._ltov(self.out_num), y)))

        # add bias
        X = self._add_bias(X)

        # initialize hidden weight vecotors at random
        n = X.shape[1]
        for m in self.hid_nums:
            np.random.seed()
            w = np.random.uniform(-1., 1., (m, n))
            self.ws.append(w)
            n = m

        # initialize output weight vecotors at random
        np.random.seed()
        w = np.random.uniform(-1., 1., (self.out_num, n))
        self.ws.append(w)

        # fitting
        for i in range(self.epochs):
            for index in range(0, self.out_num, self.batch_size):

                _x = X[index:index + self.batch_size]
                _y = y[index:index + self.batch_size]

                zs, us, deltas, dws = [], [], [], []

                z = _x.T
                zs.append(z)

                for w in self.ws[:-1]:
                    u = np.dot(w, z)
                    z = self._sigmoid(u)
                    us.append(u)
                    zs.append(z)

                delta = _y.T - self._predict(_x)
                deltas.append(delta)
                for u, w in zip(reversed(us), reversed(self.ws)):
                    delta = self._get_delta(w, delta, u)
                    deltas.append(delta)

                for delta, z in zip(reversed(deltas), zs):
                    dws.append(np.dot(delta, z.T) / self.batch_size)

                for i, dw in enumerate(dws):
                    self.ws[i] -= self.r * dw


def main():
    db_name = 'iris'
    #db_name = 'australian'

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    mlp = MLP(hid_nums=[10], epochs=10, batch_size=1)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    mlp.fit(X_train, y_train)
    re = mlp.predict(X_train)

    print(re)
    print(y_train)

    score = sum([r == y for r, y in zip(re, y_train)]) / len(y_train)
    print("general Accuracy %0.3f " % score)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
