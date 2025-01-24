import matplotlib.pyplot as plt
import numpy as np

# 原始目标函数
def g(x):
    return 0.1 * (x ** 3 + x ** 2 + x)

# 生成训练数据
train_x = np.linspace(-2, 2, 8)
train_y = g(train_x) + np.random.randn(train_x.size) * 0.05

# 测试用的 x 轴
x = np.linspace(-2, 2, 100)

# 数据标准化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    """标准化函数"""
    return (x - mu) / sigma


# 构造多项式设计矩阵
def to_matrix(x, degree=10):
    """
    将输入数据 x 转换为多项式设计矩阵
    :param x: 输入数据
    :param degree: 多项式的最高阶
    :return: 多项式设计矩阵
    """
    return np.vstack([x ** i for i in range(degree + 1)]).T


# 计算预测值
def f(x, theta):
    """线性模型预测函数"""
    return np.dot(x, theta)


# 计算均方误差
def E(x, y, theta):
    """均方误差函数"""
    return 0.5 * np.sum((y - f(x, theta)) ** 2)


# 梯度下降优化函数
def gradient_descent(X, y, theta, eta, lambda_=0, tol=1e-6):
    """
    使用梯度下降法优化参数
    :param X: 设计矩阵
    :param y: 目标值
    :param theta: 参数向量
    :param eta: 学习率
    :param lambda_: 正则化系数（默认为 0，表示普通最小二乘法）
    :param tol: 收敛阈值
    :return: 优化后的参数
    """
    diff = 1
    error = E(X, y, theta)
    while diff > tol:
        # 正则化项梯度（不对偏置项进行正则化）
        reg_term = lambda_ * np.hstack([0, theta[1:]])
        # 梯度更新
        theta = theta - eta * (np.dot(f(X, theta) - y, X) + reg_term)
        # 计算误差变化
        current_error = E(X, y, theta)
        diff = error - current_error
        error = current_error
    return theta


# 数据标准化和设计矩阵构造
train_z = standardize(train_x)
X = to_matrix(train_z)

# 初始参数
initial_theta = np.random.randn(X.shape[1])

# 超参数
ETA = 1e-4  # 学习率
LAMBDA = 1  # 正则化系数

# 普通最小二乘法（OLS）
theta_ols = gradient_descent(X, train_y, initial_theta, ETA)

# 带正则化的最小二乘法（Ridge Regression）
theta_ridge = gradient_descent(X, train_y, initial_theta, ETA, lambda_=LAMBDA)

# 绘制结果
plt.figure(figsize=(8, 6))
plt.scatter(train_z, train_y, label="Training Data", color="blue", marker="o")
z = standardize(x)  # 测试数据标准化
plt.plot(z, f(to_matrix(z), theta_ols), linestyle="dashed", label="OLS Fit", color="red")
plt.plot(z, f(to_matrix(z), theta_ridge), label="Ridge Fit", color="green")
plt.xlabel("Standardized Feature")
plt.ylabel("Target")
plt.title("Polynomial Regression: OLS vs Ridge")
plt.legend()
plt.grid()
plt.show()
