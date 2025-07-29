from flask import Flask, render_template, request, jsonify
from detector import predict_news

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('news', '')
    label, confidence = predict_news(text)
    return jsonify({'prediction': label, 'confidence': f"{confidence:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
