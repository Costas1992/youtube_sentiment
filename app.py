from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

# Loading Environment Variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YT_API_KEY")

# Model,Tokenizer Paths
MODEL_PATH = "models/sentiment_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_PATH = "models/sentiment_label.pickle"
MAX_LEN = 250

# Loading Model and Tokenizer
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

label_encoder = {0: "positive", 1: "negative"}

#Flask App
app = Flask(__name__)

# Predict Sentiment
def predict_sentiment(text_list):
    sequences = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    preds = model.predict(padded)
    class_indices = np.argmax(preds, axis=1)
    sentiments = [label_encoder[i] for i in class_indices]
    return sentiments

# Fetch YouTube Comments
def get_video_comments(video_url, max_comments=10):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        return comments
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

# Routes

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    comment = None
    if request.method == "POST":
        comment = request.form["comment"]
        sentiment = predict_sentiment([comment])[0]
    return render_template("index.html", comment=comment, sentiment=sentiment)

@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    video_url = request.form["video_url"]
    comments = get_video_comments(video_url, max_comments=10)

    if not comments:
        return render_template("video_results.html", error="No comments found or invalid video URL.")

    sentiments = predict_sentiment(comments)
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")

    result = {
        "video_url": video_url,
        "total": len(comments),
        "positive": pos,
        "negative": neg,
        "avg_score": round(pos / len(comments) * 100, 2)
    }

    return render_template("video_results.html", result=result, comments=zip(comments, sentiments))

# API endpoint for programmatic use
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    comment = data.get("comment", "")
    sentiment = predict_sentiment([comment])[0]
    return jsonify({"comment": comment, "sentiment": sentiment})

# Run App
if __name__ == "__main__":
    app.run(debug=True)