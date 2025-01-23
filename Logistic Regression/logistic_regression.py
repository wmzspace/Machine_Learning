import numpy as np
import matplotlib.pyplot as plt

# 加载数据
# 假设 CSV 文件包含三列：前两列是特征，第三列是标签
train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]  # 提取特征矩阵 (m, 2)
train_y = train[:, 2]    # 提取标签向量 (m,)

# 初始化参数
theta = np.random.rand(3)  # 随机初始化参数向量 (3,)
mu = train_x.mean(axis=0)  # 计算每列特征的均值 (用于标准化)
sigma = train_x.std(axis=0)  # 计算每列特征的标准差 (用于标准化)

# 特征标准化函数
def standardize(x):
    """
    对特征矩阵进行标准化（Z-Score 标准化）
    :param x: 原始特征矩阵
    :return: 标准化后的特征矩阵
    """
    return (x - mu) / sigma

# 标准化训练数据
train_z = standardize(train_x)

# 添加偏置项的函数
def to_matrix(x):
    """
    将特征矩阵扩展为包含偏置项的一列 1
    :param x: 标准化后的特征矩阵
    :return: 扩展后的特征矩阵 (m, n+1)
    """
    x0 = np.ones([x.shape[0], 1])  # 创建一列全为 1 的列向量
    return np.hstack([x0, x])      # 将偏置列与原特征矩阵拼接

# 扩展特征矩阵
X = to_matrix(train_z)

# 可视化原始数据（标准化后的）
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o', label='Class 1')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x', label='Class 0')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.title('Training Data')
plt.show()

# 定义 Sigmoid 激活函数
def sigmoid(x):
    """
    计算 Sigmoid 函数的值
    :param x: 输入值 (特征矩阵与参数的点积)
    :return: Sigmoid 激活后的值 (概率)
    """
    return 1 / (1 + np.exp(-np.dot(x, theta)))

# 梯度下降参数
ETA = 1e-3  # 学习率
epoch = 5000  # 最大迭代次数

# 梯度下降训练
for _ in range(epoch):
    # 计算预测值
    predictions = sigmoid(X)  # 预测概率 (m,)
    # 计算误差 (预测值 - 真实标签)
    errors = predictions - train_y  # (m,)
    # 计算梯度并更新参数
    theta -= ETA * np.dot(errors, X)  # 梯度公式：∇J(θ) = (h(X) - y) * X

# 决策边界可视化
x0 = np.linspace(-2, 2, 100)  # 生成 x0 的等间距点
decision_boundary = -(theta[0] + theta[1] * x0) / theta[2]  # 决策边界公式

# 绘制分类结果和决策边界
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o', label='Class 1')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x', label='Class 0')
plt.plot(x0, decision_boundary, linestyle='dashed', label='Decision Boundary')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.title('Decision Boundary')
plt.show()
