#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-17 09:24:24
# @Author  : LZR (sharp_l@163.com)
# @Link    : ${link}
# @Version : $Id$

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt


def demo(n, k):
    X, t = make_swiss_roll(n_samples=n, noise=1)

    le = SpectralEmbedding(n_components=2, n_neighbors=k, affinity='nearest_neighbors')
    le_X = le.fit_transform(X)

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(0.33))
    axes[0].set_axis_off()
    axes[0] = plt.subplot(121, projection='3d')
    axes[0].scatter(*X.T, c=t, s=50)
    axes[0].set_title('Swiss Roll')
    axes[1].scatter(*le_X.T, c=t, s=50)
    axes[1].set_title('LE Embedding')

    plt.show()


if __name__ == '__main__':
    demo(n=1500, k=5)
