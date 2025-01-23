"""
感知机适用于线性可分数据，输出的是离散的类别标签（+1 或 -1）。其
优化目标并不明确，主要通过误分类样本来调整权重，因此对噪声和异常值较为敏感，容易受到干扰。
在收敛性上，感知机无法处理线性不可分的数据，可能导致无法收敛。
感知机的计算复杂度较低，适合简单的二分类任务。
此外，感知机可以通过核方法扩展到非线性问题，但整体适用范围相对有限。
"""

from time import sleep

import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('images1.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

w = np.random.rand(2)


def f(x):
    return 1 if np.dot(w, x) >= 0 else -1


epoch = 10
count = 0
x1 = np.arange(0, 500)

for _ in range(epoch):
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x
            count += 1
            print(' 第 {} 次 : w = {}'.format(count, w))
            plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o', color='blue')
            plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x', color='orange')
            plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed', color='red')
            plt.plot(x[0], x[1], "*", markersize=15, color='red')
            plt.ylim([0, 500])
            plt.title('Perceptron')
            plt.show()
            sleep(0.1)
