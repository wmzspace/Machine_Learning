import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csr_matrix
from tqdm import tqdm  # 导入 tqdm 库


class LogisticRegressionFromScratch:
    def __init__(self, lr=0.01, max_iter=1000, C=1.0, class_weight=None):
        self.lr = lr
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + cp.exp(-z))

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = csr_matrix(X)
        elif isinstance(X, np.ndarray):
            X = cp.array(X, dtype=cp.float32)
        else:
            raise ValueError("Unsupported data type for X")

        if not isinstance(y, cp.ndarray):
            y = cp.array(y, dtype=cp.int32)
        elif y.dtype != cp.int32:
            y = y.astype(cp.int32)

        n_samples, n_features = X.shape
        self.weights = cp.random.randn(n_features) * 0.01
        self.bias = 0

        if self.class_weight == 'balanced':
            class_counts = cp.bincount(y)
            total_samples = len(y)
            weights = total_samples / (len(class_counts) * class_counts)
            sample_weights = cp.array([weights[label] for label in y])
        else:
            sample_weights = cp.ones_like(y, dtype=cp.float32)

        # 使用 tqdm 添加进度条，设置 leave=False 保证进度条在一行显示
        for _ in tqdm(range(self.max_iter), desc="Training Logistic Regression", leave=False):
            linear_model = X.dot(self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            error = y_pred - y
            dw = (1 / n_samples) * X.T.dot(error * sample_weights) + (1 / self.C) * self.weights
            db = (1 / n_samples) * cp.sum(error * sample_weights)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = csr_matrix(X)
        elif isinstance(X, np.ndarray):
            X = cp.array(X, dtype=cp.float32)
        else:
            raise ValueError("Unsupported data type for X")
        linear_model = X.dot(self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class OneVsRestClassifierFromScratch:
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator if base_estimator else LogisticRegressionFromScratch()
        self.models = []

    def fit(self, X, y):
        self.classes_ = y.columns if isinstance(y, pd.DataFrame) else cp.unique(y)
        self.models = []

        # 使用 tqdm 添加进度条，设置 leave=False 保证进度条在一行显示
        for cls in tqdm(self.classes_, desc="Training One-vs-Rest Classifier", leave=False):
            y_binary = (y[cls] if isinstance(y, pd.DataFrame) else (y == cls)).astype(int).values.ravel()
            y_binary = cp.array(y_binary, dtype=cp.int32)
            model = LogisticRegressionFromScratch(lr=self.base_estimator.lr, max_iter=self.base_estimator.max_iter,
                C=self.base_estimator.C, class_weight=self.base_estimator.class_weight)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        probs = cp.array([model.predict_proba(X) for model in self.models]).T
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        return cp.array([model.predict_proba(X) for model in self.models]).T
