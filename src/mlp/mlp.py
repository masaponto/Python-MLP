#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.utils import shuffle

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

    def _vtol(self, vec):
        """
        tranceform vector(numpy array) to label (integer)
        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Examples
        >>> mlp = MLP()
        >>> mlp.out_num = 3
        >>> mlp._vtol([1, -1, -1])
        1
        >>> mlp._vtol([-1, 1, -1])
        2
        >>> mlp._vtol([-1, -1, 1])
        3
        """

        if self.out_num == 1:
            return 1 if 1 == np.around(vec) else -1
        else:
            v = list(vec)
            return int(v.index(max(v))) + 1

    def _ltov(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        Exmples:
        >>> mlp = MLP(10, 3)
        >>> mlp._ltov(3, 1)
        [1, -1, -1]
        >>> mlp._ltov(3, 2)
        [-1, 1, -1]
        >>> mlp._ltov(3, 3)
        [-1, -1, 1]
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def _add_bias(self, x_vs):
        """
        add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec

        Examples:
        >>> mlp = MLP()
        >>> mlp._add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[ 1.,  2.,  3.,  1.],
               [ 1.,  2.,  3.,  1.]])
        """
        return np.c_[x_vs, np.ones(len(x_vs))]

    def _get_delta(self, w, delta, u):
        return self._dsigmoid(u) * np.dot(w.T, delta)

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

    def predict(self, x):
        x = self._add_bias(x)
        y = x.T

        for w in self.ws:
            y = self._sigmoid(np.dot(w, y))

        #print(y)
        #[print(_y) for _y in y.T]
        return np.array([self._vtol(_y) for _y in y.T])

    def fit(self, X, y):

        # initialize weights
        self.ws = []

        # number of training data
        self.data_num = X.shape[0]
        assert(self.data_num > self.batch_size)

        # numer of output neurons
        self.out_num = max(y)

        # fix data label(integer) to array
        y = np.array([self._ltov(self.out_num, _y) for _y in y])

        # add bias
        X = self._add_bias(X)

        # initialize hidden weight vecotors at random
        n = X.shape[1]
        for m in self.hid_nums:
            np.random.seed()
            self.ws.append(np.random.uniform(-1., 1., (m, n)))
            n = m

        # initialize output weight vecotors at random
        np.random.seed()
        self.ws.append(np.random.uniform(-1., 1., (self.out_num, n)))

        # fitting
        for i in range(self.epochs):
            X, y = shuffle(X, y, random_state=np.random.RandomState())
            for index in range(0, self.data_num, self.batch_size):

                end = index + self.batch_size
                end_index = end if end <= self.data_num else self.data_num

                _x = X[list(range(index, end_index)), :]
                _y = y[list(range(index, end_index))]

                zs, us, deltas, dws = [], [], [], []

                z = _x.T
                zs.append(z)

                for w in self.ws:
                    u = np.dot(w, z)
                    z = self._sigmoid(u)
                    us.append(u)
                    zs.append(z)

                delta = (z - _y.T) * self._dsigmoid(u)
                deltas.append(delta)

                for _u, _w in zip(reversed(us[:-1]), reversed(self.ws)):
                    delta = self._get_delta(_w, delta, _u)
                    deltas.append(delta)

                for _delta, _z in zip(reversed(deltas), zs[:-1]):
                    dws.append(np.dot(_delta, _z.T) / self.batch_size)

                for j, dw in enumerate(dws):
                    self.ws[j] -= self.r * dw


def main():
    #db_name = 'iris'
    db_name = 'australian'

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    mlp = MLP(hid_nums=[30], epochs=1000, batch_size=1)

    mlp.fit(data_set.data, data_set.target)
    re = mlp.predict(data_set.data)
    score = sum([r == y for r, y in zip(re, data_set.target)]
                        ) / len(data_set.target)

    print("Accuracy %0.3f " % score)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
