import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from textblob import TextBlob
import pickle
import os

# loading the dataset

df = pd.read_csv("data/combined_comments.csv")
df = df.dropna(subset=["comment"])
df = df[df["comment"].str.strip() != ""]
df = df.drop_duplicates(subset=["comment"])


def map_sentiment(comment):
    polarity = TextBlob(str(comment)).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["comment"].apply(map_sentiment)

df = df[df["sentiment"].isin(["positive", "negative"])]

positive_df = df[df["sentiment"] == "positive"]
negative_df = df[df["sentiment"] == "negative"]

if len(positive_df) > len(negative_df):
    negative_df = resample(negative_df, replace=True, n_samples=len(positive_df), random_state=42)
else:
    positive_df = resample(positive_df, replace=True, n_samples=len(negative_df), random_state=42)
df_balanced =pd.concat([positive_df, negative_df])

print("\nBalanced dataset:")
print(df_balanced["sentiment"].value_counts())
# preparing features and labels / split into training testing sets and vectorize

x = df["comment"]
y = df["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english")
x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)

# Naive Bayes

model = MultinomialNB()
model.fit(x_train_vectors, y_train)

# model evaluation and testing with custom comments

y_pred = model.predict(x_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.2f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

sample_comments = [
    "I like this video",
    "This video Sucks",
    "Amazing explanation",
    "Awful editing, waste of time."
]

sample_vectors = vectorizer.transform(sample_comments)
sample_predictions = model.predict(sample_vectors)
print("\nSample predictions:")
for comment, pred in zip(sample_comments, sample_predictions):
    print(f"{comment} -> {pred}")
    
    # saving training model
    
os.makedirs("model", exist_ok=True)

with open("models/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/tfids_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("\nTrained model and vectorizer saved in 'models/' folder")
