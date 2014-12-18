#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x, a):
    return 1 / (1 + np.exp(-a*x) )


def sigmoid_(x, a):
    return a * x * (1.0 - x)


def calc_out(w_vs, x_v, a):
    return sigmoid( np.dot(w_vs, x_v), a )


def out_error(d_v, out_v, a):
    return (out_v - d_v) * sigmoid_(out_v, a)


def mid_error(mid_v, wo_vs, eo_v, a):
    return np.dot(wo_vs.T, eo_v) * sigmoid_(mid_v, a)


def mid_w_update(wm_vs, mid_v, wo_vs, em_v, x_v, r):
    x_v = np.atleast_2d( x_v )
    em_v = np.atleast_2d( em_v )
    return wm_vs - r * np.dot(em_v.T, x_v)


def out_w_update(wo_vs, eo_v, mid_v, r):
    mid_v = np.atleast_2d( mid_v )
    eo_v = np.atleast_2d( eo_v )
    return wo_vs - r * np.dot( eo_v.T, mid_v )


def add_bius(x_vs) :
    return [ np.append(x_v, 1) for x_v in x_vs ]


def plot_dots (x_vs, d_vs):
    for i in range(len(x_vs)):
        xs = list(x_vs[i])
        x = xs[0]
        y = xs[1]

        if d_vs[i] == 1 :
            plt.plot(x, y, "r.", label="class 1")
        else :
            plt.plot(x, y, "k.", label="class 2")


def plot_func(w_v):
    xs = range(-5,5)
    ys = [-(w_v[0]/w_v[1]) * x - (w_v[2]/w_v[1]) for x in xs]
    plt.plot(xs,ys)


if __name__ == "__main__":

    # シグモイド関数の定数
    a = 1

    # 中間層ノード数
    mid_num = 2
    # 出力層ノード数
    out_num = 1

    # 学習回数
    epochs = 10000

    # 学習率
    r = 0.5

    # 特徴ベクトル
    x_vs = [np.array([1,1]), np.array([1,0]), np.array([0,1]), np.array([0,0])]

    # バイアス項追加
    x_vs = add_bius(x_vs)

    x_vd = len(x_vs[0])

    # 重み
    wm_vs = np.random.uniform(-1.0, 1.0, (mid_num, x_vd))
    wo_vs = np.random.uniform(-1.0, 1.0, (out_num, mid_num))

    #XOR
    d_vs = np.array( [0, 1, 1, 0] )
    #OR
    #d_vs = np.array( [1, 1, 1, 0] )
    #AND
    #d_vs = np.array( [1, 0, 0, 0] )

    #d_vs = np.array([ [0,0], [1,1], [1,1], [0,0] ])

    count = 0

    #学習
    while count < epochs:

        count += 1

        for i in range(len(d_vs)):

            # forward phase
            # 中間層の結果
            mid_v = calc_out(wm_vs, x_vs[i], a)
            # 出力層の結果
            out_v = calc_out(wo_vs, mid_v, a)


            # backward phase
            # 出力層の誤差
            eo_v = out_error(d_vs[i], out_v, a)
            # 中間層
            em_v = mid_error(mid_v, wo_vs, eo_v, a)


            # weight update
            #中間層
            wm_vs = mid_w_update(wm_vs, mid_v, wo_vs, em_v, x_vs[i], r)
            #出力層
            wo_vs = out_w_update(wo_vs, eo_v, mid_v, r)


    # 結果の出力
    for i in range(0, len(d_vs)):
        mid_v = calc_out(wm_vs, x_vs[i], a)
        out_v = calc_out(wo_vs, mid_v, a)
        x_v = list(x_vs[i])
        x_v.pop()

        print( str(x_v) + ":" + str(out_v) )


    #グラフ描画
    plot_dots(x_vs, d_vs)

    for wm_v in wm_vs :
        plot_func(wm_v)

    plt.show()
