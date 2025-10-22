# YouTube Sentiment Analyzer  

A simple Flask web app that uses a TensorFlow model to predict if a YouTube comment is **positive** or **negative**.

---

## Overview

This project analyzes the sentiment of YouTube comments.  
It was built with:
- **Python**
- **TensorFlow / Keras**
- **Flask**

You can train your own model or use the provided one.

---

## 🗂 Project Structure

- **youtube_sentiment/**
- **app.py**                  
- **training_tensorflow.py**  
- **models/**                 
- **templates/**              
- **requirements.txt**
- **README.md**
---

## Run Locally

```bash
git clone https://github.com/Costas1992/youtube_sentiment.git
cd youtube_sentiment
pip install -r requirements.txt
python app.py

Then open your browser at:
👉 http://127.0.0.1:5000

☁Deploy on Render
1.	Push your repo to GitHub
2.	On Render, create a new Web Service
3.	Use:
•	Build Command: pip install -r requirements.txt
•	Start Command: gunicorn app:app

Render will give you a public URL to share your app.

⸻
Author
Kostas
Self-taught developer & ML enthusiast
Currently studying Applied Machine Learning in Norway.

Project inspiration from Youtube videos
