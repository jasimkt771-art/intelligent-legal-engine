import redis
import uuid
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import (REDIS_HOST, REDIS_PORT, REDIS_DB, PINECONE_API_KEY, PINECONE_INDEX_NAME2, SEMANTIC_CACHE_THRESHOLD, REDIS_CACHE_TTL)
import re

def connect_to_redis():
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
        db=REDIS_DB, decode_responses=True)
    print("Redis connected successfully")
    return redis_client

def connect_to_cache_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME2)
    print("Semantic cache connected successfully")
    return index

def connect_to_cache():
    redis = connect_to_redis()
    semantic = connect_to_cache_index()
    return redis, semantic

def embed_query(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model.encode(query).tolist()


def check_exact_cache(redis_client, query):
    try:
        return redis_client.get(query)
    except redis.ConnectionError:
        return None

def extract_article_number(query):
    match = re.search(r'\bArticle\s+(\d+[A-Za-z]*)\b', query, re.IGNORECASE)

    if match:
        return match.group(1).lower()

    return None

def check_semantic_cache(cache_index, query):
    vector = embed_query(query)

    results = cache_index.query(
        vector=vector,
        top_k=1,
        include_metadata=True
    )

    if results.matches:
        match = results.matches[0]
        score = match.score

        cached_query = match.metadata.get("query", "")

        query_article = extract_article_number(query)
        cached_article = extract_article_number(cached_query)

        print("\nQuery:", query)
        print("Cached query:", cached_query)
        print("Semantic score:", score)
        print("Semantic Threshold: ", SEMANTIC_CACHE_THRESHOLD)
        print("Query article:", query_article)
        print("Cached article:", cached_article)

        if query_article and cached_article:
            if query_article != cached_article:
                return None

        if score >= SEMANTIC_CACHE_THRESHOLD:
            return match.metadata["response"]

    return None

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
    print('Checked redis')
    if response:
        print("Cached from redis")
        return response

    else:
        response2 = check_semantic_cache(cache_index, query)
        print('Checked semantic cache')
        if response2:
            print("Cached from semantic cache")
            return response2

def save_to_cache(redis_client, cache_index, query, response):
    save_exact_cache(redis_client, query, response)
    print("Saved to redis")
    save_semantic_cache(cache_index, query, response)
    print("Saved to semantic cache")

def clear_cache(redis_client, cache_index):
    redis_client.flushdb()
    print("Redis cleared.")
    try:
        cache_index.delete(delete_all=True)
        print("Semantic cache cleared.")
    except Exception as e:
        print("Semantic cache is already empty.")

def show_exact_cache(redis_client):
    keys = redis_client.keys("*")

    if not keys:
        print("Redis cache is empty.")
        return

    print("\n=== Redis Cache ===")

    for key in keys:
        print(f"\nQuery: {key}")
        print(f"Response: {redis_client.get(key)}")

if __name__ == "__main__":
    redis_client = connect_to_redis()
    cache_index = connect_to_cache_index()
    clear_cache(redis_client, cache_index)
    show_exact_cache(redis_client)

'''
    query = "What is Article 21?"
#    response = "Article 21 guarantees protection of life and personal liberty."
    response = check_semantic_cache(cache_index, query)

    save_to_cache(redis_client, cache_index, query, response)

    print(check_cache(redis_client, cache_index, query))

    clear_cache(redis_client, cache_index)
    print(redis_client.keys("*"))'''