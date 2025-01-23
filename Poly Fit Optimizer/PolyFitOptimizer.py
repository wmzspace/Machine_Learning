import matplotlib.pyplot as plt
import numpy as np


# 加载数据
def load_data(file_path):
    """
    加载数据并返回输入特征和目标值。
    :param file_path: 数据文件路径
    :return: 输入特征 train_x, 目标值 train_y
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]


# 标准化函数
def standardize(x):
    """
    对输入数据进行 z-score 标准化。
    :param x: 输入数组
    :return: 标准化后的数组
    """
    return (x - x.mean()) / x.std()


# 构造设计矩阵
def to_matrix(x):
    """
    将输入 x 转换为设计矩阵，包含常数项、一次项和二次项。
    :param x: 输入数组
    :return: 设计矩阵
    """
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


# 均方误差 (MSE) 计算
def mse(x, y, theta):
    """
    计算均方误差 (Mean Squared Error)。
    :param x: 输入特征矩阵
    :param y: 实际目标值
    :param theta: 参数向量
    :return: 均方误差
    """
    return np.mean((y - np.dot(x, theta)) ** 2)


# 梯度下降
def gradient_descent(x, y, theta, eta=1e-3, tolerance=1e-2):
    """
    使用随机梯度下降法 (SGD) 优化参数。
    :param x: 输入特征矩阵
    :param y: 实际目标值
    :param theta: 参数向量
    :param eta: 学习率
    :param tolerance: 误差差值阈值
    :return: 优化后的参数 theta, 误差记录 errors
    """
    errors = [mse(x, y, theta)]
    diff = float('inf')  # 初始误差差值
    while diff > tolerance:
        # 随机打乱数据
        p = np.random.permutation(x.shape[0])
        for x_i, y_i in zip(x[p, :], y[p]):
            theta -= eta * (np.dot(x_i, theta) - y_i) * x_i
        # 记录误差
        errors.append(mse(x, y, theta))
        diff = errors[-2] - errors[-1]
    return theta, errors


# 绘制图像
def plot_scatter(x, y, title, xlabel, ylabel):
    """
    绘制散点图。
    :param x: x 轴数据
    :param y: y 轴数据
    :param title: 图像标题
    :param xlabel: x 轴标签
    :param ylabel: y 轴标签
    """
    plt.scatter(x, y, color="blue", label="Data")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()


def plot_curve(x, y, theta, title):
    """
    绘制拟合曲线。
    :param x: 输入特征
    :param y: 实际目标值
    :param theta: 参数向量
    :param title: 图像标题
    """
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.dot(to_matrix(x_line), theta)
    plt.scatter(x, y, color="blue", label="Data")
    plt.plot(x_line, y_line, color="red", label="Fitted Curve")
    plt.title(title)
    plt.xlabel("Standardized X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()


def plot_errors(errors):
    """
    绘制误差下降曲线。
    :param errors: 误差记录
    """
    plt.plot(np.arange(len(errors)), errors, color="green", label="MSE")
    plt.title("Error Reduction")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.show()


# 主函数
def main():
    # 加载和预处理数据
    train_x, train_y = load_data('click.csv')
    train_z = standardize(train_x)  # 标准化输入特征
    X = to_matrix(train_z)  # 构造设计矩阵

    # 初始化参数
    theta = np.random.rand(3)  # 随机初始化 theta

    # 绘制原始数据和标准化数据
    plot_scatter(train_x, train_y, "Original Data", "train_x", "train_y")
    plot_scatter(train_z, train_y, "Standardized Data", "train_z", "train_y")

    # 梯度下降优化
    theta, errors = gradient_descent(X, train_y, theta)

    # 绘制误差下降曲线
    plot_errors(errors)

    # 绘制拟合曲线
    plot_curve(train_z, train_y, theta, "Fitted Curve")


# 执行主函数
if __name__ == "__main__":
    main()
