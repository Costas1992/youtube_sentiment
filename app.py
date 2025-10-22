from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model paths
MODEL_PATH = "models/sentiment_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
MAX_LEN = 250

# Load model and tokenizer
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Flask app
app = Flask(__name__)

def predict_sentiment(text_list):
    sequences = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    preds = model.predict(padded)
    preds = np.argmax(preds, axis=1)
    sentiments = ["negative" if p == 0 else "positive" for p in preds]
    return sentiments

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        comment = request.form["comment"]
        sentiment = predict_sentiment([comment])[0]
        return render_template("index.html", comment=comment, sentiment=sentiment)
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    comment = data.get("comment", "")
    sentiment = predict_sentiment([comment])[0]
    return jsonify({"comment": comment, "sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)