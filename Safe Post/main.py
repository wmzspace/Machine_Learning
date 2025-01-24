import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 示例数据
data = {
    "Post Text": [
        # 正面内容
        "You are an amazing friend!",
        "Good morning, hope you have a wonderful day!",
        "This is such a thoughtful and kind gesture.",
        "I really appreciate your help.",
        "You are so talented and hardworking!",
        "What a beautiful day to be alive!",
        "I love the way you think about this problem.",
        "This is the best advice I have ever received.",
        "Your creativity is truly inspiring.",
        "Thank you for always being there for me.",

        # 负面内容
        "You are the worst person I have ever met.",
        "This is absolutely disgusting and unacceptable.",
        "I can't believe how stupid you are.",
        "Your work is terrible and worthless.",
        "You should stop talking, no one likes you.",
        "This is the most offensive thing I've ever seen.",
        "You are so annoying and pathetic.",
        "I hate everything about you.",
        "You are a complete failure.",
        "This is a waste of everyone's time."
    ],
    "Label": [
        # 正面内容标签
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 负面内容标签
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(df["Post Text"], df["Label"], test_size=0.2, random_state=42)

# 特征提取：TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 模型训练：Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 预测概率
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# 评估模型
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
