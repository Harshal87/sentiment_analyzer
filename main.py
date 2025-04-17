from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enrich import enrich_result, ASTRA_DB_TOKEN, ASTRA_DB_API_ENDPOINT, COLLECTION_NAME, DataAPIClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← or restrict to ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Query(BaseModel):
    query: str

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

    enriched = [enrich_result(doc, data.query) for doc in results]
    sorted_results = sorted(enriched, key=lambda x: (x["sentiment_score"], x["similarity"]), reverse=True)
    return sorted_results[:8]
