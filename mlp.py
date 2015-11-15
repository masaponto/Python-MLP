#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file


class MLP(BaseEstimator):
    """
    Multi Layer Perceptron (Sngle hidden layer)

    """

    def __init__(self, hid_num, epochs, r=0.5, a=1):
        """
        mlp using sigmoid
        Args:
        hid_num int : number of hidden neuron
        out_num int : number of output neuron
        epochs int: number of epoch
        r float: learning rate
        a int:  sigmoid constant value
        """
        self.hid_num = hid_num
        self.epochs = epochs
        self.r = r
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return 1 / (1 + np.exp(- self.a * x))

    def _dsigmoid(self, x):
        """
        diff sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return self.a * x * (1.0 - x)

    def _calc_out(self, w_vs, x_v):
        """
        Args:
        w_vs [[float]]
        x_v [float] : feature vector
        Returns:
        float
        """

        return self._sigmoid(np.dot(w_vs, x_v))

    def _out_error(self, d_v, out_v):
        """
        出力層の誤差
        Args:
        d_v [float]
        out_v [float]
        Returns:
        [float]
        """
        return (out_v - d_v) * self._dsigmoid(out_v)

    def _mid_error(self, mid_v, eo_v):
        """
        中間層の誤差
        Args:
        mid_v  [float]
        eo_v [float]
        Returns:
        [float]
        """
        return np.dot(self.wo_vs.T, eo_v) * self._dsigmoid(mid_v)

    def _w_update(self, w_vs, e_v, i_v):
        """
        update weights
        Args:
        w_vs [[float]]
        e_v [float]
        i_v [float]
        Return:
        [[float]]
        """
        e_v = np.atleast_2d(e_v)
        i_v = np.atleast_2d(i_v)
        return w_vs - self.r * np.dot(e_v.T, i_v)

    def _add_bias(self, x_vs):
        """
        add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec
        """

        return np.c_[x_vs, np.ones(len(x_vs))]

    def _ltov(self, n):
        """
        trasform label scalar to vector

        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label

        Exmples:
        >>> p = MLP(5, 100)
        >>> p._ltov(3)(1)
        [1, -1, -1]
        >>> p._ltov(3)(2)
        [-1, 1, -1]
        >>> p._ltov(3)(3)
        [-1, -1, 1]
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

        Exmples:
        >>> p = MLP(10, 3)
        >>> p.out_num = 3
        >>> p._vtol([1, -1, -1])
        1
        >>> p._vtol([-1, 1, -1])
        2
        >>> p._vtol([-1, -1, 1])
        3
        """
        fix_num = lambda x: 1 if 1 == round(x, 0) else -1
        if self.out_num == 1:
            return fix_num(vec[0])
        else:
            v = list(vec)
            return int(v.index(max(v))) + 1


    def fit(self, X, y):
        """
        学習
        Args:
        X [[float]] array : featur vector
        y [int] array : class labels
        """

        self.out_num = max(y)

        if self.out_num == 1:
            d_vs = y
        else:
            d_vs = np.array(list(map(self._ltov(self.out_num), y)))

        x_vs = self._add_bias(X)
        x_vd = len(x_vs[0])

        # 重み
        np.random.seed()
        self.wm_vs = np.random.uniform(-1.0, 1.0, (self.hid_num, x_vd))
        self.wo_vs = np.random.uniform(-1.0, 1.0, (self.out_num, self.hid_num))

        for n in range(self.epochs):
            for d_v, x_v in zip(d_vs, x_vs):

                # forward phase
                # 中間層の結果
                mid_v = self._calc_out(self.wm_vs, x_v)
                # 出力層の結果
                out_v = self._calc_out(self.wo_vs, mid_v)

                # backward phase
                # 出力層の誤差
                eo_v = self._out_error(d_v, out_v)
                # 中間層
                em_v = self._mid_error(mid_v,  eo_v)

                # weight update
                # 中間層
                self.wm_vs = self._w_update(self.wm_vs, em_v, x_v)

                # 出力層
                self.wo_vs = self._w_update(self.wo_vs, eo_v, mid_v)


    def predict(self, x_vs):
        """
        Args:
        x_vs [[float]] array
        """
        x_vs = self._add_bias(x_vs)
        mid_vs = [self._calc_out(self.wm_vs, x_v) for x_v in x_vs]
        out_vs = [self._vtol(self._calc_out(self.wo_vs, mid_v))
                  for mid_v in mid_vs]
        return np.array(out_vs)

    def one_predict(self, x_v):
        """
        Args:
        x_v [float] array
        """
        x_v = np.append(x_v, 1)
        mid_v = self._calc_out(self.wm_vs, x_v)
        out_v = self._calc_out(self.wo_vs, mid_v)

        return self._vtol(out_v)


def main():
    db_names = ['iris', 'australian']
    hid_nums = [10, 20, 30, 40]

    for db_name in db_names:
        print(db_name)
        # load iris data set
        data_set = fetch_mldata(db_name)
        data_set.data = preprocessing.scale(data_set.data)

        print('MLP')
        for hid_num in hid_nums:
            print(str(hid_num), end=' ')
            mlp = MLP(5, 100)
            ave = 0

            for i in range(3):
                scores = cross_validation.cross_val_score(
                    mlp, data_set.data, data_set.target, cv=5, scoring='accuracy', n_jobs=-1)
                ave += scores.mean()
            ave /= 3
            print("Accuracy: %0.3f " % (ave))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
