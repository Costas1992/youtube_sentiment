from googleapiclient.discovery import build
import pandas as pd
from dotenv import load_dotenv
import os
import string
from textblob import TextBlob

# load .env automatically

load_dotenv()

API_KEY = os.getenv("YT_API_KEY")

youtube = build('youtube', 'v3', developerKey=API_KEY)


# get comment function

def get_video_comments(video_id, max_comments=500):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()
    
    while response and len(comments) < max_comments:
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        
        if "nextPageToken" in response and len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response["nextPageToken"],
                maxResults=100,
                textFormat="plainText"
            ).execute()
        else:
            break
    
    return comments


# clean function

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# automate label function

def label_sentiment(comment):
    polarity = TextBlob(comment).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    else:
        return "neutral"
    
    # some video ids to grab comments


video_ids = [
    "w4rG5GY9IlA",
    "HAnw168huqA",
    "eHJnEHyyN1Y",
    "ewpGDPft7Mo",
    "G1p6rlDCxq0",
    "Xzv84ZdtlE0"
]


# Grab clean and label

all_comments = []

for vid in video_ids:
    raw_comments = get_video_comments(vid, max_comments=500)
    for c in raw_comments:
        clean_comment = clean_text(c)
        sentiment = label_sentiment(clean_comment)
        all_comments.append((clean_comment, clean_comment, sentiment))

# Create Data frame

df = pd.DataFrame(all_comments, columns=["comment", "sentiment", "video_id"])
df.to_csv("comments.csv", index=False)
df.to_csv("data/combined_comments.csv", index=False)

print("Saved combined_comments.csv with", len(df), "comments")
print(df.head())
