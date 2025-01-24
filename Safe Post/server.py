import joblib
from flask import Flask, request, render_template, jsonify
from googletrans import Translator

app = Flask(__name__)

# 标签与中文对应
LABELS = {"toxic": "恶意", "severe_toxic": "严重恶意", "obscene": "淫秽", "threat": "威胁", "insult": "侮辱",
          "identity_hate": "身份仇恨"}


def translate_and_detect_language(text):
    """
    使用 Googletrans 翻译文本并检测语言。
    如果是中文，则翻译为英文；如果是英文，则翻译为中文。
    """
    translator = Translator()
    try:
        # 检测语言
        detected_lang = translator.detect(text)
        if detected_lang.lang == 'zh-CN':  # 如果是中文，翻译为英文
            translation = translator.translate(text, src='zh-cn', dest='en')
            return translation.text, f"翻译为英文: {translation.text}"
        else:  # 如果是其他语言，翻译为中文
            translation = translator.translate(text, src='en', dest='zh-cn')
            return text, f"翻译为中文: {translation.text}"
    except Exception as e:
        print(f"Error translating text: {e}")
        return text, "翻译失败"


def load_model_and_predict(model_filename, vectorizer_filename, new_data):
    """
    加载保存的模型和向量化器，并对新数据进行预测。
    """
    try:
        # 加载模型和向量化器
        model = joblib.load(model_filename)
        vectorizer = joblib.load(vectorizer_filename)

        # 转换新数据为特征向量
        new_data_tfidf = vectorizer.transform([new_data])

        # 预测概率
        predictions = model.predict_proba(new_data_tfidf)
        return predictions[0]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.json.get("text")  # 从 JSON 数据中获取文本
        if not input_text:
            return jsonify({"error": "请输入文本！"}), 400

        # 翻译和语言检测
        processed_text, translation = translate_and_detect_language(input_text)

        # 加载模型和预测
        model_filename = "./model/model.joblib"
        vectorizer_filename = "./model/vectorizer.joblib"
        predictions = load_model_and_predict(model_filename, vectorizer_filename, processed_text)

        if predictions is None:
            return jsonify({"error": "无法加载模型或向量化器文件！"}), 500

        # 格式化预测结果
        threshold = 0.5
        formatted_results = [(LABELS[label], prob) for label, prob in zip(LABELS.keys(), predictions) if
                             prob >= threshold]

        # 如果没有任何异常内容，返回默认提示
        if not formatted_results:
            formatted_results = [("无异常内容", 1.0)]  # 默认显示“无异常内容”

        return jsonify({"translation": translation, "results": formatted_results})
    return render_template("index.html")


def main():
    app.run(debug=True, port=5002, host="0.0.0.0")


if __name__ == "__main__":
    main()
