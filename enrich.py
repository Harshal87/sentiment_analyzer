# enrich.py
from astrapy import DataAPIClient
import requests
from difflib import SequenceMatcher
from geopy.distance import geodesic

# Replace these with your actual API keys
ASTRA_DB_TOKEN = "AstraCS:LOZryRCyfyqCeJpzAfWZYsAA:ff54edbe919c68c7b2c8b236ebd3ba0c8c2cef55313fb381ab5d8958ddf97045"
ASTRA_DB_API_ENDPOINT = "https://fdc90b78-019f-4b61-8c87-a56bba9e070e-us-east-2.apps.astra.datastax.com"
COLLECTION_NAME = "zomato_pune_1"  # Replace with your collection name
HF_API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
HF_TOKEN = "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"  # Replace with your Hugging Face API token
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def compute_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_sentiment(user_input, tags):
    sentence = f"{user_input} - {tags}"
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": sentence}, timeout=10)
        result = response.json()

        if not result:
            return "Unknown", 0

        label = result[0][0]["label"]
        score = round(result[0][0]["score"] * 100, 2)
        
        # If sentiment is always neutral, handle gracefully
        if score == 0:
            return "Neutral", 50

        sentiment = "Positive" if "4" in label or "5" in label else "Negative" if "1" in label or "2" in label else "Neutral"
        return sentiment, score
    except Exception as e:
        print(f"Sentiment API failed: {e}")
        return "Unknown", 50  # Default fallback sentiment

def get_coords_from_area(area):
    try:
        access_key = "661bbebf20123d87bb4f82711450e4b2"  # Replace with your PositionStack API key
        url = f"http://api.positionstack.com/v1/forward?access_key={access_key}&query={area},Pune"
        response = requests.get(url).json()
        if response["data"]:
            lat = response["data"][0]["latitude"]
            lon = response["data"][0]["longitude"]
            return lat, lon
        return None
    except:
        return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def enrich_result(doc, query, area=None):
    tags = doc.get("reviews_1", "")
    similarity = round(compute_similarity(query, tags) * 100, 2)
    sentiment, sentiment_score = get_sentiment(query, tags)
    doc["similarity"] = similarity
    doc["sentiment"] = sentiment
    doc["sentiment_score"] = sentiment_score

    # Get user coordinates from the area
    user_coords = None
    if area:
        user_coords = get_coords_from_area(area)

    # Get restaurant coordinates
    if "latitude" in doc and "longitude" in doc:
        rest_coords = (float(doc["latitude"]), float(doc["longitude"]))
        if user_coords:
            doc["distance_km"] = round(haversine(user_coords[0], user_coords[1], rest_coords[0], rest_coords[1]), 2)
        else:
            doc["distance_km"] = None
    else:
        doc["distance_km"] = None

    return doc
