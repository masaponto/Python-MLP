#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt


class MLP:

    def __init__(self, mid_num, out_num, epochs, r, a):
        """mlp using sigmoid
        mid_num: 中間層ノード数
        out_num: 出力層ノード数
        epochs: 学習回数
        r: 学習率
        a: シグモイド関数の定数
        """
        self.mid_num = mid_num
        self.out_num = out_num
        self.epochs = epochs
        self.r = r
        self.a = a

    def sigmoid(self, x, a):
        return 1 / (1 + np.exp(-a * x))

    def sigmoid_(self, x, a):
        """シグモイド関数微分"""
        return a * x * (1.0 - x)

    def calc_out(self, w_vs, x_v, a):
        return self.sigmoid(np.dot(w_vs, x_v), a)

    def out_error(self, d_v, out_v, a):
        """出力層の誤差"""
        return (out_v - d_v) * self.sigmoid_(out_v, a)

    def mid_error(self, mid_v, wo_vs, eo_v, a):
        """中間層の誤差"""
        return np.dot(wo_vs.T, eo_v) * self.sigmoid_(mid_v, a)

    def w_update(self, w_vs, e_v, i_v, r):
        """重み更新"""
        e_v = np.atleast_2d(e_v)
        i_v = np.atleast_2d(i_v)
        return w_vs - r * np.dot(e_v.T, i_v)

    def add_bias(self, x_v):
        """バイアス項追加"""
        return np.append(x_v, 1)

    def fit(self, X, y):
        """学習"""
        x_vs = X
        d_vs = y

        x_vs = list(map(self.add_bias, x_vs))
        x_vd = len(x_vs[0])

        # 重み
        self.wm_vs = np.random.uniform(-1.0, 1.0, (self.mid_num, x_vd))
        self.wo_vs = np.random.uniform(-1.0, 1.0, (self.out_num, self.mid_num))

        for n in range(self.epochs):
            for i in range(len(d_vs)):

                # forward phase
                # 中間層の結果
                mid_v = self.calc_out(self.wm_vs, x_vs[i], self.a)
                # 出力層の結果
                out_v = self.calc_out(self.wo_vs, mid_v, self.a)

                # backward phase
                # 出力層の誤差
                eo_v = self.out_error(d_vs[i], out_v, self.a)
                # 中間層
                em_v = self.mid_error(mid_v, self.wo_vs, eo_v, self.a)

                # weight update
                # 中間層
                self.wm_vs = self.w_update(self.wm_vs, em_v, x_vs[i], self.r)

                # 出力層
                self.wo_vs = self.w_update(self.wo_vs, eo_v, mid_v, self.r)

    def predict(self, x_v):
        x_v = self.add_bias(x_v)
        mid_v = self.calc_out(self.wm_vs, x_v, self.a)
        out_v = self.calc_out(self.wo_vs, mid_v, self.a)
        return out_v


def plot(mlp, x_vs, y_vs):
    """plot function
    """
    for i in range(len(x_vs)):
        x_v = x_vs[i]
        x = x_v[0]
        y = x_v[1]

        if y_vs[i] == 1:
            plt.plot(x, y, "r.", label="class 1")
        else:
            plt.plot(x, y, "k.", label="class 2")

    for w_v in mlp.wm_vs:
        xs = range(-5, 5)
        ys = [-(w_v[0] / w_v[1]) * x - (w_v[2] / w_v[1])
              for x in xs]
        plt.plot(xs, ys)

    plt.savefig("graph.png")


if __name__ == "__main__":

    # data
    X_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y_train = [1, 0, 0, 1]

    mid_num = 2
    out_num = 1
    epochs = 10000
    r = 0.5
    a = 1

    mlp = MLP(mid_num, out_num, epochs, r, a)
    mlp.fit(X_train, y_train)

    plot(mlp, X_train, y_train)

    result = list(map(mlp.predict,X_train))
    for i in range(len(result)):
        print(str(result[i]) + ":" + str(y_train[i]))
