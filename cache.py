import redis
import uuid
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import (REDIS_HOST, REDIS_PORT, REDIS_DB, PINECONE_API_KEY, PINECONE_INDEX_NAME2, SEMANTIC_CACHE_THRESHOLD, REDIS_CACHE_TTL)

def connect_to_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
        db=REDIS_DB, decode_responses=True)

def connect_to_cache_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME2)

def embed_query(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model.encode(query).tolist()


def check_exact_cache(redis_client, query):
    return redis_client.get(query)

def check_semantic_cache(cache_index, query):
    vector = embed_query(query)

    results = cache_index.query(
        vector=vector,
        top_k=1,
        include_metadata=True
    )

    if results.matches:
        score = results.matches[0].score
        print(f"Similarity Score: {score}")

        if score >= SEMANTIC_CACHE_THRESHOLD:
            return (
                results.matches[0].metadata["response"],
                score
            )

    return None, None

def save_exact_cache(redis_client, query, response):
    redis_client.set(
        query,
        response,
        ex=REDIS_CACHE_TTL
    )

def save_semantic_cache(cache_index, query, response):
    vector = embed_query(query)

    cache_index.upsert(
        vectors=[
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {
                    "query": query,
                    "response": response
                }
            }
        ]
    )

def check_cache(redis_client, cache_index, query):
    response = check_exact_cache(redis_client, query)

    if response:
        return response

    return check_semantic_cache(cache_index, query)

def save_to_cache(redis_client, cache_index, query, response):
    save_exact_cache(redis_client, query, response)
    save_semantic_cache(cache_index, query, response)

def clear_cache(redis_client, cache_index):
    redis_client.flushdb()

    cache_index.delete(
        delete_all=True
    )

if __name__ == "__main__":
    redis_client = connect_to_redis()
    cache_index = connect_to_cache_index()

    query = "What is Article 21?"
#    response = "Article 21 guarantees protection of life and personal liberty."
    response = check_semantic_cache(cache_index, query)
'''
    save_to_cache(redis_client, cache_index, query, response)

    print(check_cache(redis_client, cache_index, query))

    clear_cache(redis_client, cache_index)
    print(redis_client.keys("*"))'''