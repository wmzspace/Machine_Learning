""" 线性可分示例
逻辑回归适用于线性可分和线性不可分数据，输出的是样本属于某一类别的概率值（范围在 0 到 1 之间）。
其优化目标是最大化对数似然函数，相较于感知机，逻辑回归对噪声和异常值更鲁棒，模型更稳定。
在收敛性方面，即使数据线性不可分，逻辑回归也能通过概率分布找到最优解。
由于使用梯度下降或其他优化方法，逻辑回归的计算复杂度较高。
此外，逻辑回归需要通过特征转换来处理非线性问题，但其适用范围更广，可用于二分类、多分类和概率预测任务。
"""

import numpy as np
import matplotlib.pyplot as plt

# 加载数据
# 假设 CSV 文件包含三列：前两列是特征，第三列是标签
train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]  # 提取特征矩阵 (m, 2)
train_y = train[:, 2]  # 提取标签向量 (m,)

# 初始化参数
theta = np.random.rand(3)  # 随机初始化参数向量 (3,)
mu = train_x.mean(axis=0)  # 计算每列特征的均值 (用于标准化)
sigma = train_x.std(axis=0)  # 计算每列特征的标准差 (用于标准化)


# 特征标准化函数
def standardize(x):
    """对特征矩阵进行标准化（Z-Score 标准化）"""
    return (x - mu) / sigma


# 添加偏置项的函数
def to_matrix(x):
    """将特征矩阵扩展为包含偏置项的一列 1"""
    x0 = np.ones([x.shape[0], 1])  # 创建一列全为 1 的列向量
    return np.hstack([x0, x])  # 将偏置列与原特征矩阵拼接


# 定义 Sigmoid 激活函数
def sigmoid(x):
    """计算 Sigmoid 函数的值"""
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# 绘制决策边界函数
def plot_decision_boundary(train_z, train_y, theta, iteration=None):
    """
    绘制分类结果和决策边界
    :param train_z: 标准化后的特征矩阵
    :param train_y: 标签
    :param theta: 当前模型参数
    :param iteration: 当前迭代次数（用于标题显示）
    """
    # 计算决策边界
    x_min, x_max = train_z[:, 0].min(), train_z[:, 0].max()  # 获取 x0 的范围
    x0 = np.array([x_min, x_max])  # 只取数据点的最小值和最大值
    decision_boundary = -(theta[0] + theta[1] * x0) / theta[2]  # 决策边界公式

    # 绘图
    plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o', label='Class 1')
    plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x', label='Class 0')
    plt.plot(x0, decision_boundary, linestyle='dashed', label='Decision Boundary')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    if iteration is not None:
        plt.title(f'Logistic Regression (Iteration {iteration})')
    else:
        plt.title('Logistic Regression')
    plt.legend()
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.show()


# 数据预处理
train_z = standardize(train_x)  # 标准化训练数据
X = to_matrix(train_z)  # 扩展特征矩阵

# 可视化原始数据
plot_decision_boundary(train_z, train_y, theta, iteration=0)

# 梯度下降参数
ETA = 1e-3  # 学习率
epoch = 2500  # 最大迭代次数

# 梯度下降训练
for i in range(epoch):
    # 计算预测值
    predictions = sigmoid(X)  # 预测概率 (m,)
    # 计算误差 (预测值 - 真实标签)
    errors = predictions - train_y  # (m,)
    # 计算梯度并更新参数
    theta -= ETA * np.dot(errors, X)  # 梯度公式：∇J(θ) = (h(X) - y) * X

    # 每 50 次迭代绘制一次决策边界
    if i % 50 == 0 or i == epoch - 1:
        plot_decision_boundary(train_z, train_y, theta, iteration=i + 1)

