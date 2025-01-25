import joblib
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# 标签与中文对应
LABELS = {"toxic": "恶意", "severe_toxic": "严重恶意", "obscene": "淫秽", "threat": "威胁", "insult": "侮辱",
          "identity_hate": "身份仇恨"}

# 百度翻译 API 配置
API_KEY = "MU5qXT0i0j3EyxsWH16MGgnQ"  # 替换为你的百度翻译 API Key
SECRET_KEY = "plwhV5XMxw6P53m0VoD4uDXevFHZu5FZ"  # 替换为你的百度翻译 Secret Key


def get_access_token():
    """
    获取百度翻译 API 的 Access Token
    """
    url = f"https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY, }
    response = requests.get(url, params=params)
    result = response.json()
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Failed to get access token: {result}")


def translate_and_detect_language(text):
    """
    使用百度翻译 API 翻译文本并检测语言。
    如果是中文，则翻译为英文；如果是英文，则翻译为中文。
    """
    try:
        # 获取 Access Token
        access_token = get_access_token()
        url = f"https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token={access_token}"

        # 检测语言（简单实现：根据字符判断是否为中文）
        def is_chinese(text):
            for char in text:
                if '\u4e00' <= char <= '\u9fff':
                    return True
            return False

        # 设置翻译方向
        from_lang = "zh" if is_chinese(text) else "en"
        to_lang = "en" if from_lang == "zh" else "zh"

        # 构造请求数据
        headers = {"Content-Type": "application/json"}
        payload = {"q": text, "from": from_lang, "to": to_lang, }

        # 发送 POST 请求
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()

        # 检查返回结果
        if "result" in result and "trans_result" in result["result"]:
            translated_text = result["result"]["trans_result"][0]["dst"]
            if from_lang == "zh":
                return translated_text, f"翻译为英文: {translated_text}"
            else:
                return text, f"翻译为中文: {translated_text}"
        else:
            error_msg = result.get("error_msg", "未知错误")
            print(f"Error from Baidu API: {error_msg}")
            return text, "翻译失败"
    except Exception as e:
        print(f"Error translating text: {e}")
        return text, "翻译失败"


def load_model_and_predict(model_filename, vectorizer_filename, scaler_filename, new_data):
    """
    加载保存的模型和向量化器，并对新数据进行预测。
    """
    try:
        # 加载模型和向量化器
        model = joblib.load(model_filename)
        vectorizer = joblib.load(vectorizer_filename)
        scaler = joblib.load(scaler_filename)

        new_data = [new_data]

        # 转换新数据为特征向量
        new_data_tfidf = vectorizer.transform(new_data)

        new_data_tfidf = scaler.transform(new_data_tfidf)

        # 预测概率
        predictions = model.predict_proba(new_data_tfidf)
        return predictions
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model and vectorizer files exist.")
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
        scaler_filename = "./model/scaler.joblib"
        predictions = load_model_and_predict(model_filename, vectorizer_filename, scaler_filename, processed_text)

        if predictions is None:
            return jsonify({"error": "无法加载模型或向量化器文件！"}), 500

        # 格式化预测结果
        threshold = 0.5
        formatted_results = [(LABELS[label], float(prob)) for label, prob in zip(LABELS.keys(), predictions[0]) if
                             prob >= threshold]

        # 如果没有任何异常内容，返回默认提示
        if not formatted_results:
            formatted_results = [("无异常内容", 1.0)]  # 默认显示“无异常内容”

        return jsonify({"translation": translation, "results": formatted_results})
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5003, host="0.0.0.0")
