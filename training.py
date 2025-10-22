import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# create a sample data for training


data = {
   "comment_text": [
       "I love this video!",
       "This tutorial is terrible.",
       "Amazing content! very helpful.",
       "I hate this channel.",
       "Fantastic explanation, thanks",
       "Worst thing I have seen.",
       "This sucks.",
       "Best tutorial",
   ],
    "sentiment": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "negative",
        "positive",

    ]
}

df = pd.DataFrame(data)
print(df)

# clean text function

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["coment_text"] = df["comment_text"].apply(clean_text)

# vectorize with CountVectorizer and after chang it to TfidfVectorizer
# Convert text data to numerical features

vectorizer = TfidfVectorizer(stop_words="english")
x = vectorizer.fit_transform(df['comment_text'])
y = df['sentiment']

print("Features names:", vectorizer.get_feature_names_out())
print("Feature matrix shape:", x.shape)

# splitting the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Naive Bayer classifier training

model = MultinomialNB()
model.fit(x_train, y_train)

# model evaluation

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Actual labels:", y_test.values)
print("Model accuracy:", accuracy)

# new comments

new_comments = [
    "I really love this video!",
    "This was the worst tutorial ever.",
    "Fantastic job explaining this concept.",
    "I am disappointed, it was boring.",
    "Amazing content; very helpful.",
    "I hate this channel.",

]

new_x = vectorizer.transform(new_comments)
predictions = model.predict(new_x)

for comment_text, sentiment in zip(new_comments, predictions):
    print(f"{comment_text} -> {sentiment}")
