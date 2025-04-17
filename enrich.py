from astrapy import DataAPIClient
import requests
from difflib import SequenceMatcher

# === CONFIG ===
ASTRA_DB_TOKEN = "your-astra-token"
ASTRA_DB_API_ENDPOINT = "your-db-endpoint"
COLLECTION_NAME = "zomato_pune_1"
HF_API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
HF_TOKEN = "your-huggingface-token"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def compute_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_sentiment(user_input, tags):
    sentence = f"{user_input} - {tags}"
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": sentence}, timeout=10)
        result = response.json()
        label = result[0][0]["label"]
        score = round(result[0][0]["score"] * 100, 2)
        sentiment = "Positive" if "4" in label or "5" in label else "Negative" if "1" in label or "2" in label else "Neutral"
        return sentiment, score
    except:
        return "Unknown", 0

def enrich_result(doc, query):
    tags = doc.get("reviews_1", "")
    similarity = round(compute_similarity(query, tags) * 100, 2)
    sentiment, sentiment_score = get_sentiment(query, tags)
    doc["similarity"] = similarity
    doc["sentiment"] = sentiment
    doc["sentiment_score"] = sentiment_score
    return doc
