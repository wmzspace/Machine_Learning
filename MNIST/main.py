import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 配置 matplotlib 显示中文和负号
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')

# 查看部分训练集图片
def plot_sample_images(x, y, num_images=64):
    fig, axes = plt.subplots(int(np.sqrt(num_images)), int(np.sqrt(num_images)), figsize=(8, 8))
    for ax in axes.flat:
        idx = np.random.randint(0, x.shape[0])
        ax.imshow(x[idx], cmap='gray')
        ax.set_title(y[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_images(x_train, y_train)

# 数据预处理：添加通道维度并归一化
x_train = x_train[..., np.newaxis] / 255.0  # (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis] / 255.0    # (10000, 28, 28, 1)

print(f'x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}')

# 创建 CNN 模型
def create_cnn_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),  # 输入层，指定输入形状
        Conv2D(32, kernel_size=(3, 3), activation='relu', name='Conv1'),  # 第一卷积层
        MaxPooling2D(pool_size=(2, 2), name='Pool1'),  # 第一池化层
        Conv2D(64, kernel_size=(3, 3), activation='relu', name='Conv2'),  # 第二卷积层
        MaxPooling2D(pool_size=(2, 2), name='Pool2'),  # 第二池化层
        Flatten(name='Flatten'),  # 展平层
        Dense(128, activation='relu', name='Dense1'),  # 全连接层
        Dense(10, activation='linear', name='Output')  # 输出层（未激活，使用 softmax 激活在损失函数中处理）
    ], name='CNN_Model')
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.001),  # 固定学习率
        metrics=['accuracy']  # 添加准确率指标
    )
    return model

model_cnn = create_cnn_model()
model_cnn.summary()

# 训练模型
history = model_cnn.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1)

# 模型性能评估
test_loss, test_accuracy = model_cnn.evaluate(x_test, y_test, verbose=0)
print(f"测试集损失值: {test_loss}")
print(f"测试集准确率: {test_accuracy * 100:.2f}%")

# 使用测试集预测
z_test_hat = model_cnn.predict(x_test)
y_test_hat = np.argmax(tf.nn.softmax(z_test_hat).numpy(), axis=1)

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
    disp.plot(cmap='viridis')
    plt.title("测试集混淆矩阵")
    plt.show()

plot_confusion_matrix(y_test, y_test_hat)

# 绘制训练损失曲线
def plot_loss_curve(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label="训练损失")
    plt.plot(history.history['accuracy'], label="训练准确率")
    plt.xlabel("训练轮次")
    plt.ylabel("值")
    plt.title("训练损失与准确率曲线")
    plt.legend()
    plt.grid()
    plt.show()

plot_loss_curve(history)

# 可视化部分预测结果
def plot_predictions(x, y_true, y_pred, num_images=16):
    fig, axes = plt.subplots(int(np.sqrt(num_images)), int(np.sqrt(num_images)), figsize=(8, 8))
    for ax in axes.flat:
        idx = np.random.randint(0, x.shape[0])
        ax.imshow(x[idx].squeeze(), cmap='gray')  # 去掉通道维度
        ax.set_title(f"预测: {y_pred[idx]}\n真实: {y_true[idx]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_predictions(x_test, y_test, y_test_hat)
