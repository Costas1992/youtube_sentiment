import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# dataset loading

df = pd.read_csv('data/combined_comments.csv')
df = df.dropna(subset=["comment"])
df = df[df["comment"].str.strip() != ""]
df = df.drop(subset=["comment"])

# use textblob

def map_sentiment(comment):
    polarity = TextBlob(str(comment)).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"
    
df["sentiment"] = df["comment"].apply(map_sentiment)

print(f"Dataset size: {len(df)}")
print(df["sentiment"].value_counts())

# encoding labels

label_encoder = LabelEncoder()
df["sentiment"] = label_encoder.fit_transform(df["sentiment"])
# 0 = negative 1 = positive
num_classes = len(label_encoder.classes_)

# tokenize
VOCAB_SIZE = 10000
MAX_LEN = 250

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="OOV")
tokenizer.fit_on_texts(df["comment"])
sequences = tokenizer.texts_to_sequences(df["comment"])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

# train split

x_train, x_test, y_train, y_test = train_test_split(padded_sequences, df["sentiment_encoded"], test_size=0.2, random_state=42)


# LSTM architecture  building the tensorflow

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary

# train and evaluate

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=64,
    epochs=10,
)

# save the model and tokenizer

os.makedirs("saved_models", exist_ok=True)
model.save("models/sentiment_model.keras")
with open("models/tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
with open("models/encoder.pickle", "wb") as f:
    pickle.dump(VOCAB_SIZE, f)
    
print("Model, tokenizer and label encoder saved in "models/" folder.)

# predictionn function

def predict_sentiment(text_list):
    with (open("models/tokenizer.pickle", "rb")) as handle:
        tokenizer_loaded = pickle.load(handle)
    with (open("models/encoder.pickle", "rb")) as handle:
        le_loaded = pickle.load(handle)
    model_loaded = tf.keras.models.load_model("models/sentiment_model.keras")
    
    sequences = tokenizer_loaded.texts_to_sequences(text_list)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    
    preds = model_loaded.predict(padded)
    class_index = np.argmax(preds, axis=1)
    sentiments = le_loaded.inverse_transform(class_index)
    return sentiments


sample_comments = [
    "I like this video",
    "This video sucks",
    "Amazing explanation",
    "Awful editing, waste of time"
]

preds = predict_sentiment(sample_comments)
for c, p in zip(sample_comments, preds):
    print(f"{c}: {p}")
