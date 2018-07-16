#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-12 08:58:53
# @Author  : LZR (sharp_l@163.com)
# @Link    : ${link}
# @Version : $Id$

from numpy import shape, mat, zeros, array, random, argsort, tile
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

Axes3D


def laplaEigen(dataMat, k, t):

    m, n = shape(dataMat)

    W = mat(zeros([m, m]))

    D = mat(zeros([m, m]))

    for i in range(m):

        k_index = knn(dataMat[i, :], dataMat, k)

        for j in range(k):

            sqDiffVector = dataMat[i, :]-dataMat[k_index[j], :]

            sqDiffVector = array(sqDiffVector)**2

            sqDistances = sqDiffVector.sum()

            W[i, k_index[j]] = math.exp(-sqDistances/t)

            D[i, i] += W[i, k_index[j]]

    L = D-W

    Dinv = np.linalg.inv(D)

    X = np.dot(D.I, L)

    lamda, f = np.linalg.eig(X)

    return lamda, f


def knn(inX, dataSet, k):

    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = array(diffMat)**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()

    return sortedDistIndicies[0:k]


def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):

    # Generate a swiss roll dataset.

    t = 1.5 * np.pi * (1 + 2 * random.rand(1, n_samples))

    x = t * np.cos(t)

    y = 83 * random.rand(1, n_samples)

    z = t * np.sin(t)

    X = np.concatenate((x, y, z))

    X += noise * random.randn(3, n_samples)

    X = X.T

    t = np.squeeze(t)

    return X, t


dataMat, color = datasets.samples_generator.make_swiss_roll(n_samples=500)
# print(color.shape)

lamda, f = laplaEigen(dataMat, 11, 5.0)

fm, fn = shape(f)

# print('fm, fn:', fm, fn)

lamdaIndicies = argsort(lamda)

first = 0

second = 0

# print(lamdaIndicies[0], lamdaIndicies[1])

for i in range(fm):

    if lamda[lamdaIndicies[i]].real > 1e-5:

        # print(lamda[lamdaIndicies[i]])

        first = lamdaIndicies[i]

        second = lamdaIndicies[i+1]

        break

print(first, second)

# redEigVects = f[:, lamdaIndicies]
Y = np.concatenate((f[:, first], f[:, second]), axis=1)
# print(Y.shape)
# print(color.shape)
# print(Y)
x, y = f[:, first].T, f[:, second].T

x=x.tolist()
y=y.tolist()
x=array(x[0])
y=array(y[0])
# print(x.shape,y.shape)
_, axes = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(0.33))
axes[0].set_axis_off()
axes[0] = plt.subplot(121, projection='3d')
axes[0].scatter(*dataMat.T, c=color, s=50)
axes[0].set_title('Swiss Roll')
axes[1].scatter(x,y, c=color, s=50)
axes[1].set_title('LE Embedding')
plt.show()
