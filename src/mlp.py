#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle


class MLP(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 hid_nums=[100],
                 epochs=1000,
                 r=0.5,
                 batch_size=20):

        self.hid_nums = hid_nums
        self.epochs = epochs
        self.r = r
        self.batch_size = batch_size

    def __ltov(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        Exmples:
        """
        return [0 if i != label else 1 for i in range(1, n + 1)]

    def __add_bias(self, x_vs):
        """
        add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec

        Examples:
        >>> mlp = MLP()
        >>> mlp._MLP__add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[ 1.,  2.,  3.,  1.],
               [ 1.,  2.,  3.,  1.]])
        """
        return np.c_[x_vs, np.ones(x_vs.shape[0])]

    def __get_delta(self, w, delta, u):
        return self.__dsigmoid(u) * np.dot(w.T, delta)

    def __sigmoid(self, x, a=1):
        """
        sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return 1 / (1 + np.exp(-a * x))

    def __dsigmoid(self, x, a=1):
        """
        derivative of sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return a * self.__sigmoid(x) * (1.0 - self.__sigmoid(x))

    def predict(self, x):
        x = self.__add_bias(x)
        y = x.T

        for w in self.Ws:
            y = self.__sigmoid(np.dot(w, y))

        if self.out_num == 1:
            y = y.T
            y[y < 0.5] = -1
            y[y >= 0.5] = 1
            return y
        else:
            return np.argmax(y.T, 1) + np.ones(y.T.shape[0])

    def fit(self, X, y):

        # initialize weights
        self.Ws = []

        # number of training data
        self.data_num = X.shape[0]
        assert(self.data_num > self.batch_size)

        # numer of output neurons
        self.out_num = max(y)

        # fix data label(integer) to array
        y = np.array([self.__ltov(self.out_num, _y) for _y in y])

        # add bias
        X = self.__add_bias(X)

        # initialize hidden weight vecotors at random
        n = X.shape[1]
        for m in self.hid_nums:
            np.random.seed()
            self.Ws.append(np.random.uniform(-1., 1., (m, n)))
            n = m

        # initialize output weight vecotors at random
        np.random.seed()
        self.Ws.append(np.random.uniform(-1., 1., (self.out_num, n)))

        # fitting
        for i in range(self.epochs):
            X, y = shuffle(X, y, random_state=np.random.RandomState())
            for index in range(0, self.data_num, self.batch_size):

                end = index + self.batch_size
                end_index = end if end <= self.data_num else self.data_num

                _x = X[index:end_index]
                _y = y[index:end_index]

                zs, us, deltas, dws = [], [], [], []

                z = _x.T
                zs.append(z)

                for w in self.Ws:
                    u = np.dot(w, z)
                    z = self.__sigmoid(u)
                    us.append(u)
                    zs.append(z)

                # add bias
                for i in range(1, len(zs) - 1):
                    zs[i][-1] = np.full((1, zs[i].shape[1]), -1.)

                delta = (z - _y.T) * self.__dsigmoid(u)
                deltas.append(delta)

                for _u, _w in zip(reversed(us[:-1]), reversed(self.Ws)):
                    delta = self.__get_delta(_w, delta, _u)
                    deltas.append(delta)

                for _delta, _z in zip(reversed(deltas), zs[:-1]):
                    dws.append(np.dot(_delta, _z.T) / self.batch_size)

                for j, dw in enumerate(dws):
                    self.Ws[j] -= self.r * dw

        return self


def main():

    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split

    db_name = 'iris'

    data_set = fetch_mldata(db_name)
    data_set.data = normalize(data_set.data)

    mlp = MLP(hid_nums=[5], epochs=1000, batch_size=1)

    X_train, X_test, y_train, y_test = train_test_split(
        data_set.data, data_set.target, test_size=0.4)

    mlp.fit(X_train, y_train)

    print("Accuracy %0.3f " % mlp.score(X_test, y_test))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
