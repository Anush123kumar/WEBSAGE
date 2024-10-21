import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, Index
from pyngrok import ngrok
import logging
import uvicorn
import nest_asyncio

# Necessary for Colab or environments that don't support asyncio natively
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load sentence transformer model (for embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Milvus (Assuming Milvus is running locally on default port 19530)
connections.connect("default", host="localhost", port="19530")

# Define the collection schema for storing vectors
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_model.get_sentence_embedding_dimension()),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]

collection_name = "text_embedding_collection"

# Create or load the collection in Milvus
try:
    collection = Collection(name=collection_name)
    logger.info(f"Collection '{collection_name}' already exists.")
except Exception as e:
    schema = CollectionSchema(fields, description="Text embedding collection")
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"Collection '{collection_name}' created.")

# Create an index on the embedding field for faster search
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
Index(collection, "embedding", index_params)
logger.info(f"Index created on collection '{collection_name}'.")

# Load the collection for querying and inserting data
collection.load()

# Pydantic models to structure the API input
class LoadDataRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

# Web scraping function to extract data from a URL
def extract_data(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page, status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all('p')
    text = "  ".join([para.get_text() for para in paragraphs])
    return text

# Function to embed text and store it into Milvus
def embed_and_store_text(text: str):
    sentences = text.split('.')
    embeddings = embedding_model.encode(sentences).tolist()
    entities = [embeddings, sentences]
    collection.insert(entities)

# Function to generate text using Gemini API
def generate_text_with_gemini(prompt: str) -> str:
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    api_key = "AIzaSyAQgnzV8fuLIRMC20arnS0zHT_y82e0aco"  # Replace with your actual API key
    full_url = f"{api_url}?key={api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,  # Adjust the creativity level of the response
            "maxOutputTokens": 100  # Limit the number of tokens in the response
        }
    }

    response = requests.post(full_url, headers=headers, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        candidates = response_data.get('candidates', [])
        if candidates:
            return candidates[0].get('content', {}).get('parts', [{}])[0].get('text', 'No content generated')
        else:
            return "No content generated"
    else:
        raise Exception(f"Failed to generate content: {response.status_code} - {response.text}")

# Endpoint to load data from a URL and store embeddings in Milvus
@app.post("/load", response_model=dict)
async def load_data(request: LoadDataRequest):
    try:
        content = extract_data(request.url)
        embed_and_store_text(content)
        return {"message": "Data loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to query Milvus and generate a response using Gemini
@app.post("/query", response_model=dict)
async def query_data(request: QueryRequest):
    try:
        if collection.num_entities == 0:
            raise HTTPException(status_code=400, detail="No data loaded")

        query_embedding = embedding_model.encode([request.query]).tolist()[0]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 50}
        }
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text"]
        )

        # Retrieve the most similar sentences
        best_sentences = [hit.entity.get("text") for hit in results[0]]

        # Generate a response using Gemini API
        gemini_response = generate_text_with_gemini(request.query)
        
        return {
            "query": request.query,
            "best_match_sentences": best_sentences,
            "gemini_response": gemini_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start Ngrok tunnel to expose the local FastAPI app to the internet
ngrok.set_auth_token("2mpMMMcpt0HrlH6AwvZs65KHZEe_3fwX3HE6UTZDp6Z7cNVBm")  # Set your Ngrok auth token
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
