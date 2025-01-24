import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier


# 加载数据
def load_data():
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    test_labels = pd.read_csv('dataset/test_labels.csv')
    return train, test, test_labels


# 数据预处理
def preprocess_data(train, test, test_labels):
    """
    预处理数据，移除未评分的测试样本。
    """
    # 移除未评分的测试集样本
    valid_test_ids = test_labels[test_labels.iloc[:, 1:].sum(axis=1) != -4]['id']
    test = test[test['id'].isin(valid_test_ids)]

    # 提取特征和标签
    X_train = train['comment_text']
    y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    X_test = test['comment_text']
    return X_train, y_train, X_test, test['id']


# 特征提取
def extract_features(X_train, X_test, max_features=5000):
    """
    使用 TfidfVectorizer 提取文本特征。
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


# 模型训练
def train_model(X_train_tfidf, y_train):
    """
    使用 OneVsRestClassifier 包装 LogisticRegression 进行多标签分类。
    """
    model = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5))
    model.fit(X_train_tfidf, y_train)
    return model


# 保存模型
def save_model(model, vectorizer, model_filename='model.joblib', vectorizer_filename='vectorizer.joblib'):
    """
    保存训练好的模型和向量化器到本地文件。
    """
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    print(f"Model saved to {model_filename} and vectorizer saved to {vectorizer_filename}.")


# 加载模型并预测
def load_model_and_predict(model_filename, vectorizer_filename, new_data):
    """
    加载保存的模型和向量化器，并对新数据进行预测。
    """
    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)

    # 转换新数据为特征向量
    new_data_tfidf = vectorizer.transform(new_data)

    # 预测概率
    predictions = model.predict_proba(new_data_tfidf)
    return predictions


# 输出分类指标
def print_classification_report(y_true, y_pred, categories):
    """
    打印分类报告，包括 precision、recall 和 f1-score。
    """
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=categories, zero_division=1)
    print(report)


# 绘制混淆矩阵
def plot_multilabel_confusion_matrix(y_true, y_pred, categories):
    """
    绘制多标签混淆矩阵。
    """
    cm = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 创建子图
    for i, ax in enumerate(axes.flatten()):
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix for {categories[i]}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


# 绘制 ROC 曲线
def plot_roc_curves(y_true, y_pred_proba, categories):
    """
    绘制每个类别的 ROC 曲线。
    """
    plt.figure(figsize=(10, 8))
    for i, category in enumerate(categories):
        fpr, tpr, _ = roc_curve(y_true[category], y_pred_proba[:, i])
        auc_score = roc_auc_score(y_true[category], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f'{category} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Categories')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


# 预测与提交
def predict_and_submit(model, X_test_tfidf, test_ids, output_file='submission.csv'):
    """
    预测测试集并生成提交文件。
    """
    y_pred_proba = model.predict_proba(X_test_tfidf)
    submission = pd.DataFrame(y_pred_proba,
                              columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    submission.insert(0, 'id', test_ids)
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


# 主函数
def main():
    # 加载数据
    train, test, test_labels = load_data()

    # 数据预处理
    X_train, y_train, X_test, test_ids = preprocess_data(train, test, test_labels)
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # 特征提取
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)

    # 模型训练
    model = train_model(X_train_tfidf, y_train)

    # 预测概率
    y_pred_proba = model.predict_proba(X_train_tfidf)

    # 调整阈值（可选）
    y_pred = (y_pred_proba >= 0.5).astype(int)  # 使用默认阈值 0.5 二值化

    # 输出分类指标
    print_classification_report(y_train, y_pred, categories)

    # 绘制混淆矩阵
    plot_multilabel_confusion_matrix(y_train, y_pred, categories)

    # 绘制 ROC 曲线
    plot_roc_curves(y_train, y_pred_proba, categories)

    # 询问是否保存模型
    save_choice = input("Do you want to save the model? (yes/no): ").strip().lower()
    if save_choice == 'yes':
        save_model(model, vectorizer)

    # 预测与提交
    predict_and_submit(model, X_test_tfidf, test_ids)


# 运行主函数
if __name__ == "__main__":
    main()
