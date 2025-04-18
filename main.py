# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from enrich import enrich_result, ASTRA_DB_TOKEN, ASTRA_DB_API_ENDPOINT, COLLECTION_NAME, DataAPIClient

app = FastAPI()

class Query(BaseModel):
    query: str
    area: str

@app.post("/search")
def search_restaurants(data: Query):
    client = DataAPIClient(ASTRA_DB_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
    collection = db.get_collection(COLLECTION_NAME)

    results = collection.find(
        filter={"Ratings_out_of_5": {"$gt": 3.5}},
        sort={"$vectorize": data.query},
        limit=30
    )

    enriched = [enrich_result(doc, data.query, data.area) for doc in results]

    sorted_results = sorted(
        enriched,
        key=lambda x: (-x["sentiment_score"], x.get("distance_km", float("inf")))
    )

    return sorted_results[:8]
