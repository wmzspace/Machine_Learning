import joblib
import argparse
from googletrans import Translator

# 标签与中文对应
LABELS = {
    "toxic": "恶意",
    "severe_toxic": "严重恶意",
    "obscene": "淫秽",
    "threat": "威胁",
    "insult": "侮辱",
    "identity_hate": "身份仇恨"
}

def translate_and_detect_language(texts):
    """
    使用 Googletrans 翻译文本并检测语言。
    如果是中文，则翻译为英文；如果是英文，则翻译为中文。
    """
    translator = Translator()
    translations = []
    processed_texts = []  # 存储供模型预测的英文文本
    for text in texts:
        try:
            # 检测语言
            detected_lang = translator.detect(text)
            if detected_lang.lang == 'zh-CN':  # 如果是中文，翻译为英文
                translation = translator.translate(text, src='zh-cn', dest='en')
                processed_texts.append(translation.text)  # 翻译后的英文文本
                translations.append(f"翻译为英文: {translation.text}")
            else:  # 如果是其他语言，翻译为中文
                translation = translator.translate(text, src='en', dest='zh-cn')
                processed_texts.append(text)  # 保留原始英文文本
                translations.append(f"翻译为中文: {translation.text}")
        except Exception as e:
            processed_texts.append(text)  # 如果翻译失败，保留原始文本
            translations.append("翻译失败")
            print(f"Error translating text: {e}")
    return processed_texts, translations

def load_model_and_predict(model_filename, vectorizer_filename, new_data):
    """
    加载保存的模型和向量化器，并对新数据进行预测。
    """
    try:
        # 加载模型和向量化器
        model = joblib.load(model_filename)
        vectorizer = joblib.load(vectorizer_filename)

        # 转换新数据为特征向量
        new_data_tfidf = vectorizer.transform(new_data)

        # 预测概率
        predictions = model.predict_proba(new_data_tfidf)
        return predictions
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model and vectorizer files exist.")
        return None

def format_predictions(predictions, labels, threshold=0.5):
    """
    将预测概率值与对应标签进行格式化，并以中文输出。
    只显示概率超过阈值的标签。
    """
    formatted_results = []
    for probs in predictions:
        # 将标签和概率配对，并按概率从高到低排序
        sorted_results = sorted(
            [(LABELS[label], prob) for label, prob in zip(labels, probs)],
            key=lambda x: x[1],
            reverse=True
        )
        # 筛选出概率超过阈值的标签
        filtered_results = [(label, prob) for label, prob in sorted_results if prob >= threshold]
        formatted_results.append(filtered_results)
    return formatted_results

def display_results(new_data, translations, formatted_results):
    """
    使用自然语言描述预测结果，并添加翻译信息。
    """
    for i, (text, translation, result) in enumerate(zip(new_data, translations, formatted_results)):
        print(f"\n文本 {i + 1}: {text}")
        print(f"{translation}")
        if result:
            print("预测结果 (可能性较大)：")
            for label, prob in result:
                print(f"  - {label}: 预测概率为 {prob:.2%}")  # 百分比格式显示概率
        else:
            print("预测结果：无异常")  # 当所有概率都低于阈值时
        print("-" * 50)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Load a trained model and predict on new data.")
    parser.add_argument('--model', type=str, default="./model/model.joblib", help="Path to the saved model file (default: ./model/model.joblib).")
    parser.add_argument('--vectorizer', type=str, default="./model/vectorizer.joblib", help="Path to the saved vectorizer file (default: ./model/vectorizer.joblib).")
    parser.add_argument('--data', type=str, nargs='+', required=True, help="New text data for prediction. Pass one or more strings.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Probability threshold for displaying predictions (default: 0.5).")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    model_filename = args.model
    vectorizer_filename = args.vectorizer
    new_data = args.data
    threshold = args.threshold

    # 翻译文本并检测语言
    processed_texts, translations = translate_and_detect_language(new_data)

    # 加载模型并预测
    predictions = load_model_and_predict(model_filename, vectorizer_filename, processed_texts)

    # 输出预测结果
    if predictions is not None:
        formatted_results = format_predictions(predictions, LABELS.keys(), threshold)
        display_results(new_data, translations, formatted_results)

# 运行主函数
if __name__ == "__main__":
    main()
