#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file


class MLP(BaseEstimator):

    def __init__(self, mid_num, epochs, r=0.5, a=1):
        """mlp using sigmoid
        mid_num: 中間層ノード数
        out_num: 出力層ノード数
        epochs: 学習回数
        r: 学習率
        a: シグモイド関数の定数
        """
        self.mid_num = mid_num
        self.epochs = epochs
        self.r = r
        self.a = a

    def sigmoid(self, x):
        return 1 / (1 + np.exp(- self.a * x))

    def sigmoid_(self, x):
        """シグモイド関数微分"""
        return self.a * x * (1.0 - x)

    def calc_out(self, w_vs, x_v):
        return self.sigmoid(np.dot(w_vs, x_v))

    def out_error(self, d_v, out_v):
        """出力層の誤差"""
        return (out_v - d_v) * self.sigmoid_(out_v)

    def mid_error(self, mid_v, eo_v):
        """中間層の誤差"""
        return np.dot(self.wo_vs.T, eo_v) * self.sigmoid_(mid_v)

    def w_update(self, w_vs, e_v, i_v):
        """重み更新"""
        e_v = np.atleast_2d(e_v)
        i_v = np.atleast_2d(i_v)
        return w_vs - self.r * np.dot(e_v.T, i_v)

    def add_bias(self, x_vs):
        """add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec

        """

        return np.c_[x_vs, np.ones(len(x_vs))]

    def fit(self, X, y):
        """学習"""

        self.out_num = max(y)

        if self.out_num == 1:
            d_vs = y
        else:
            d_vs = np.array(list(map(self._ltov(self.out_num), y)))

        x_vs = self.add_bias(X)
        x_vd = len(x_vs[0])

        # 重み
        np.random.seed()
        self.wm_vs = np.random.uniform(-1.0, 1.0, (self.mid_num, x_vd))
        self.wo_vs = np.random.uniform(-1.0, 1.0, (self.out_num, self.mid_num))

        for n in range(self.epochs):
            for d_v, x_v in zip(d_vs, x_vs):

                # forward phase
                # 中間層の結果
                mid_v = self.calc_out(self.wm_vs, x_v)
                # 出力層の結果
                out_v = self.calc_out(self.wo_vs, mid_v)

                # backward phase
                # 出力層の誤差
                eo_v = self.out_error(d_v, out_v)
                # 中間層
                em_v = self.mid_error(mid_v,  eo_v)

                # weight update
                # 中間層
                self.wm_vs = self.w_update(self.wm_vs, em_v, x_v)

                # 出力層
                self.wo_vs = self.w_update(self.wo_vs, eo_v, mid_v)

    def _ltov(self, n):
        """trasform label scalar to vector

            Args:
            n (int) : number of class, number of out layer neuron
            label (int) : label

            Exmples:
            >>> e = ELM(10, 3)
            >>> e._ltov(3)(1)
            [1, -1, -1]
            >>> e._ltov(3)(2)
            [-1, 1, -1]
            >>> e._ltov(3)(3)
            [-1, -1, 1]

            """
        def inltov(label):
            return [-1 if i != label else 1 for i in range(1, n + 1)]
        return inltov

    def __vtol(self, vec):
        """tranceform vector (list) to label

        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        """

        fix_num = lambda x: 1 if 1 == round(x, 0) else -1

        if self.out_num == 1:
            return fix_num(vec[0])
        else:
            v = list(map(fix_num, list(vec)))
            if len(v) == 1:
                return vec[0]
            elif (max(v) == -1):
                return 0
            return int(v.index(1)) + 1

    def predict(self, x_vs):
        x_vs = self.add_bias(x_vs)
        mid_vs = [self.calc_out(self.wm_vs, x_v) for x_v in x_vs]
        out_vs = [self.__vtol(self.calc_out(self.wo_vs, mid_v))
                  for mid_v in mid_vs]
        return np.array(out_vs)

    def one_predict(self, x_v):
        x_v = np.append(x_v, 1)

        print(x_v)

        mid_v = self.calc_out(self.wm_vs, x_v)
        out_v = self.calc_out(self.wo_vs, mid_v)
        print(out_v)

        return self.__vtol(out_v)


if __name__ == "__main__":

    #db_names = ['australian']
    db_names = ['iris']

    hid_nums = [10]

    #data_set = fetch_mldata(db_name)
    #data_set.data = preprocessing.scale(data_set.data)
    #
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #    data_set.data, data_set.target, test_size=0.4, random_state=0)
    #
    #mlp = MLP(5, 1000)
    #mlp.fit(X_train, y_train)

    #print(y_test[0])

    #print(mlp.one_predict(X_test[0]))
    #print(y_test == mlp.predict(X_test))

    #print(sum(y_test == mlp.predict(X_test)) / len(y_test))


    for db_name in db_names:
        print(db_name)
        # load iris data set
        data_set = fetch_mldata(db_name)
        data_set.data = preprocessing.scale(data_set.data)

        print('MLP')
        for hid_num in hid_nums:
            print(str(hid_num), end=' ')
            mlp = MLP(5,1000)
            ave = 0

            for i in range(10):
                scores = cross_validation.cross_val_score(
                    mlp, data_set.data, data_set.target, cv=5, scoring='accuracy', n_jobs=-1)
                ave += scores.mean()
            ave /= 10
            print("Accuracy: %0.2f " % (ave))
