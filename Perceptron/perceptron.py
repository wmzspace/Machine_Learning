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
            plt.show()
            sleep(0.1)
