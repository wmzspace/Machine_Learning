""" 线性不可分示例
逻辑回归适用于线性可分和线性不可分数据，输出的是样本属于某一类别的概率值（范围在 0 到 1 之间）。
其优化目标是最大化对数似然函数，相较于感知机，逻辑回归对噪声和异常值更鲁棒，模型更稳定。
在收敛性方面，即使数据线性不可分，逻辑回归也能通过概率分布找到最优解。
由于使用梯度下降或其他优化方法，逻辑回归的计算复杂度较高。
此外，逻辑回归需要通过特征转换来处理非线性问题，但其适用范围更广，可用于二分类、多分类和概率预测任务。
"""

import matplotlib.pyplot as plt
import numpy as np

# 加载数据
train = np.loadtxt('data3.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]  # 特征
train_y = train[:, 2]  # 标签

# 初始化参数
theta = np.random.rand(4)  # 参数向量，包含偏置和多项式系数
mu = train_x.mean(axis=0)  # 特征均值，用于标准化
sigma = train_x.std(axis=0)  # 特征标准差，用于标准化


# 数据标准化
def standardize(x):
    return (x - mu) / sigma


# 构造设计矩阵，加入二次特征
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])  # 偏置项
    x3 = x[:, 0, np.newaxis] ** 2  # 二次特征
    return np.hstack([x0, x, x3])


# Sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 线性模型
def f(x):
    return np.dot(x, theta)


# 梯度更新函数
def update_theta(X, y, theta, eta):
    predictions = sigmoid(f(X))  # 预测值
    gradient = np.dot(predictions - y, X)  # 梯度计算
    return theta - eta * gradient  # 参数更新


# 标准化训练数据并构造设计矩阵
train_z = standardize(train_x)
X = to_matrix(train_z)

# 超参数设置
ETA = 1e-3  # 学习率
EPOCHS = 5000  # 训练轮数
accuracies = []


# 分类函数
def classify(X):
    """
    根据模型的线性输出 f(X)，将预测值转换为类别 0 或 1。
    """
    predictions = sigmoid(f(X))  # 计算每个样本属于类别 1 的概率
    return (predictions >= 0.5).astype(int)  # 概率 >= 0.5 则分类为 1，否则分类为 0


p = np.random.permutation(X.shape[0])  # 随机打乱数据
batch_size = 32  # 每个小批次的大小
X_shuffled, y_shuffled = X[p], train_y[p]

# 训练逻辑回归模型
for epoch in range(EPOCHS):
    # 随机梯度下降更新参数 (SGD)
    for x, y in zip(X[p, :], train_y[p]):
        theta = update_theta(x, y, theta, ETA)

    # 批量梯度下降(Batch GD)
    # theta = update_theta(X, train_y, theta, ETA)  # 批量梯度下降更新参数

    # 小批量梯度下降 (Mini-Batch Gradient Descent)
    # for i in range(0, X.shape[0], batch_size):
    #     X_batch = X_shuffled[i:i + batch_size]
    #     y_batch = y_shuffled[i:i + batch_size]
    #     theta = update_theta(X_batch, y_batch, theta, ETA)  # 使用小批次更新

    result = classify(X) == train_y
    accuracy = np.mean(result)
    accuracies.append(accuracy)

    # 每10轮绘制一次决策边界
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        x1 = np.linspace(-2, 2, 100)  # x1的取值范围
        x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]  # 决策边界
        plt.figure()
        plt.scatter(train_z[train_y == 1, 0], train_z[train_y == 1, 1], label='Class 1', marker='o')
        plt.scatter(train_z[train_y == 0, 0], train_z[train_y == 0, 1], label='Class 0', marker='x')
        plt.plot(x1, x2, linestyle='dashed', color='red', label='Decision Boundary')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Logistic Regression (Non-linearly Separable) Epoch {epoch + 1}')
        plt.legend()
        plt.show()

x = np.arange(len(accuracies))
plt.plot(x, accuracies)
plt.show()
