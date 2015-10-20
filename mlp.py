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

        d_vs = y
        self.out_num = max(y)

        x_vs = self.add_bias(X)
        x_vd = len(x_vs[0])

        # 重み
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

    def __vtol(self, vec):
        """tranceform vector (list) to label

        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Exmples:
        >>> e = ELM(10, 3)
        >>> e.out_num = 3
        >>> e._ELM__vtol([1, -1, -1])
        1
        >>> e._ELM__vtol([-1, 1, -1])
        2
        >>> e._ELM__vtol([-1, -1, 1])
        3
        >>> e._ELM__vtol([-1, -1, -1])
        0

        """

        if self.out_num == 1:
            return round(vec[0], 0)
        else:
            v = list(vec)
            if len(v) == 1:
                return vec[0]
            elif (max(v) == -1):
                return 0
            return int(v.index(1)) + 1


    def predict(self, x_vs):
        x_vs = self.add_bias(x_vs)
        mid_vs = [self.calc_out(self.wm_vs, x_v) for x_v in x_vs]
        out_vs = [ 1 if 1 == self.__vtol( self.calc_out( self.wo_vs, mid_v) ) else -1 for mid_v in mid_vs]
        return np.array(out_vs)


    def one_predict(self, x_v):
        x_v = np.append(x_v, 1)

        print(x_v)

        mid_v = self.calc_out(self.wm_vs, x_v)
        out_v = self.calc_out( self.wo_vs, mid_v)
        print(out_v)

        #return 1 if round(out_v) == 1 else -1

        return 1 if self.__vtol(out_v)  == 1 else -1


if __name__ == "__main__":

    db_name = 'australian'
    #db_name = 'iris'

    hid_nums = [10]

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    mlp = MLP(10,100)
    mlp.fit(X_train, y_train)

    #print(y_test[0])

    #print(mlp.one_predict(X_test[0]))
    #print(y_test == mlp.predict(X_test))

    print(sum(y_test == mlp.predict(X_test)) / len(y_test) )




#    for db_name in db_names:
#        print(db_name)
#        # load iris data set
#        data_set = fetch_mldata(db_name)
#        data_set.data = preprocessing.scale(data_set.data)
#
#        print('MLP')
#        for hid_num in hid_nums:
#            print(str(hid_num), end=' ')
#            mlp = MLP(10,100)
#            ave = 0
#
#            for i in range(3):
#                scores = cross_validation.cross_val_score(
#                    mlp, data_set.data, data_set.target, cv=5, scoring='accuracy')
#                ave += scores.mean()
#            ave /= 3
#            print("Accuracy: %0.2f " % (ave))
