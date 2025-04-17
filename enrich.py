from astrapy import DataAPIClient
import requests
from difflib import SequenceMatcher

# === CONFIG ===
ASTRA_DB_TOKEN = "AstraCS:LOZryRCyfyqCeJpzAfWZYsAA:ff54edbe919c68c7b2c8b236ebd3ba0c8c2cef55313fb381ab5d8958ddf97045"
ASTRA_DB_API_ENDPOINT = "https://fdc90b78-019f-4b61-8c87-a56bba9e070e-us-east-2.apps.astra.datastax.com"
COLLECTION_NAME = "zomato_pune_1"
HF_API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
HF_TOKEN = "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"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def compute_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_sentiment(user_input, tags):
    sentence = f"{user_input} - {tags}"
    try:
        response = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": sentence},
            timeout=15  # increase timeout
        )

        try:
            result = response.json()
        except Exception as json_err:
            print("❌ Failed to parse JSON:", json_err)
            print("📥 Raw Response Text:", response.text)
            return "Unknown", 0

        if isinstance(result, dict) and "error" in result:
            print("⚠️ Hugging Face API Error:", result)
            return "Error", 0

        print("✅ HF Response:", result)
        label = result[0][0]["label"]
        score = round(result[0][0]["score"] * 100, 2)
        sentiment = (
            "Positive" if "4" in label or "5" in label
            else "Negative" if "1" in label or "2" in label
            else "Neutral"
        )
        return sentiment, score

    except requests.exceptions.RequestException as e:
        print("⚠️ HF Request Exception:", e)
        return "Unknown", 0


def enrich_result(doc, query):
    tags = doc.get("reviews_1", "")
    similarity = round(compute_similarity(query, tags) * 100, 2)
    sentiment, sentiment_score = get_sentiment(query, tags)
    doc["similarity"] = similarity
    doc["sentiment"] = sentiment
    doc["sentiment_score"] = sentiment_score
    return doc
