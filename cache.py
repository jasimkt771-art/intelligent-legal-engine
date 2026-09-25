import redis
import uuid
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_USERNAME,
    REDIS_PASSWORD,
    REDIS_DB,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME2,
    SEMANTIC_CACHE_THRESHOLD,
    REDIS_CACHE_TTL,
    REDIS_LOCAL_HOST
)
import re

# Handles exact and semantic response caching.
# Uses Redis for exact matches and Pinecone for semantically similar queries.

# Exact cache uses Redis; semantic cache uses a separate Pinecone index.
embedding_model = None
redis_client = None
semantic_cache_index = None


def connect_to_redis():
    global redis_client

    try:
        if redis_client is None:
            redis_client = redis.Redis(
                host=REDIS_LOCAL_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                username=REDIS_USERNAME,
                password=REDIS_PASSWORD,
                decode_responses=True
            )

        return redis_client

    except Exception as e:
        print(f"Error connecting to Redis(connect_to_redis[cache.py]):\n{e}")
        raise


def connect_to_semantic_cache():
    global semantic_cache_index

    try:
        if semantic_cache_index is None:
            pc = Pinecone(api_key=PINECONE_API_KEY)

            semantic_cache_index = pc.Index(PINECONE_INDEX_NAME2)

        return semantic_cache_index

    except Exception as e:
        print(f"Error connecting to semantic cache: {e}")
        raise


def connect_to_cache():
    try:
        redis_client = connect_to_redis()

    except Exception as e:
        print(
            f"Redis cache unavailable"
            f"(connect_to_cache[cache.py]): \n{e}"
        )
        redis_client = None

    try:
        semantic = connect_to_semantic_cache()

    except Exception as e:
        print(
            f"Semantic cache unavailable"
            f"(connect_to_cache[cache.py]): \n{e}"
        )
        semantic = None

    return redis_client, semantic


def embed_query(query):
    global embedding_model

    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if embedding_model is None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        embedding = embedding_model.encode(query)

        return embedding.tolist()

    except Exception as e:
        print(f"Error embedding query(embed_query[cache.py]):\n{e}")
        raise


def check_exact_cache(redis_client, query):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if redis_client is None:
            print("Redis cache unavailable. Skipping exact cache check.")
            return None

        return redis_client.get(query)

    except TypeError as e:
        print(
            f"Invalid query"
            f"(check_exact_cache[cache.py]): \n{e}"
        )
        raise

    except redis.ConnectionError as e:
        print(
            f"Redis connection error while checking exact cache"
            f"(check_exact_cache[cache.py]): \n{e}"
        )
        return None

    except Exception as e:
        print(
            f"Error checking exact cache"
            f"(check_exact_cache[cache.py]): \n{e}"
        )
        raise


def extract_article_number(query):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        match = re.search(
            r'\bArticle\s+(\d+[A-Za-z]*)\b',
            query,
            re.IGNORECASE
        )

        if match:
            return match.group(1).lower()

        return None

    except TypeError as e:
        print(
            f"Invalid query for article extraction"
            f"(extract_article_number[cache.py]): \n{e}"
        )
        raise


def check_semantic_cache(cache_index, query):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if cache_index is None:
            print(
                "Semantic cache unavailable. "
                "Skipping semantic cache check."
            )
            return None

        vector = embed_query(query)

        results = cache_index.query(
            vector=vector,
            top_k=1,
            include_metadata=True
        )

        if results.matches:
            match = results.matches[0]
            score = match.score

            try:
                cached_query = match.metadata.get("query", "")
                cached_response = match.metadata["response"]

            except KeyError as e:
                print(f"Missing required semantic cache field: \n{e}")
                raise

            query_article = extract_article_number(query)
            cached_article = extract_article_number(cached_query)

            print("\nQuery:", query)
            print("\nCached Query:\n", cached_query)
            print("\nSemantic score:", score)
            print("Semantic Threshold: ", SEMANTIC_CACHE_THRESHOLD)
            print("Query article:", query_article)
            print("Cached article:", cached_article)

            # Prevent semantically similar queries about different Articles
            # from returning the wrong cached response.
            if query_article and cached_article:
                if query_article != cached_article:
                    return None

            if score >= SEMANTIC_CACHE_THRESHOLD:
                return cached_response

        return None

    except TypeError as e:
        print(
            'Invalid Query'
            '(check_semantic_cache[cache.py]): \n',
            e
        )
        raise

    except Exception as e:
        print(
            f"Error checking semantic cache"
            f"(check_semantic_cache[cache.py]): \n{e}"
        )
        return None


def save_exact_cache(redis_client, query, response):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if redis_client is None:
            print(
                "Redis cache unavailable. "
                "Skipping exact cache save."
            )
            return None

        redis_client.set(
            query,
            response,
            ex=REDIS_CACHE_TTL
        )

    except TypeError as e:
        print(
            'Invalid Query'
            '(save_exact_cache[cache.py]): \n',
            e
        )
        raise

    except redis.ConnectionError as e:
        print(
            f"Redis connection error while saving exact cache"
            f"(save_exact_cache[cache.py]): \n{e}"
        )
        return None

    except Exception as e:
        print(
            f"Error saving exact cache"
            f"(save_exact_cache[cache.py]): \n{e}"
        )
        raise


def save_semantic_cache(cache_index, query, response):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if cache_index is None:
            print(
                "Semantic cache unavailable. "
                "Skipping semantic cache save."
            )
            return None

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

    except TypeError as e:
        print(
            'Invalid Query'
            '(save_semantic_cache[cache.py]): \n',
            e
        )
        raise

    except Exception as e:
        print(
            f"Error saving semantic cache"
            f"(save_semantic_cache[cache.py]): \n{e}"
        )
        raise


def check_cache(redis_client, cache_index, query):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        # Check exact matches first because they are cheaper and unambiguous.
        response = check_exact_cache(redis_client, query)

        print("Checked redis")

        if response:
            print("Cached from redis")
            return response

        response2 = check_semantic_cache(cache_index, query)

        print("Checked semantic cache")

        if response2:
            print("Cached from semantic cache")
            return response2

        return None

    except TypeError as e:
        print(
            'Invalid Query'
            '(check_cache[cache.py]): \n',
            e
        )
        raise

    except Exception as e:
        print(
            f"Error checking cache"
            f"(check_cache[cache.py]): \n{e}"
        )
        raise


def save_to_cache(redis_client, cache_index, query, response):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        save_exact_cache(redis_client, query, response)

        print("Saved to redis")

        save_semantic_cache(cache_index, query, response)

        print("Saved to semantic cache")

    except TypeError as e:
        print(
            'Invalid Query'
            '(save_to_cache[cache.py]): \n',
            e
        )
        raise

    except Exception as e:
        print(
            f"Error saving to cache"
            f"(save_to_cache[cache.py]): \n{e}"
        )
        raise


def clear_cache(redis_client, cache_index):
    # Clear the local Redis cache.
    try:
        if redis_client is not None:
            redis_client.flushdb()
            print("Local Redis cleared.")
        else:
            print(
                "Local Redis unavailable. "
                "Skipping local Redis clear."
            )

    except redis.ConnectionError as e:
        print(
            f"Redis connection error while clearing local cache"
            f"(clear_cache[cache.py]): \n{e}"
        )

    # Clear the Redis Cloud cache.
    try:
        redis_cloud_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            username=REDIS_USERNAME,
            password=REDIS_PASSWORD,
            decode_responses=True
        )

        redis_cloud_client.flushdb()
        print("Redis Cloud cleared.")

    except redis.ConnectionError as e:
        print(
            f"Redis connection error while clearing Redis Cloud"
            f"(clear_cache[cache.py]): \n{e}"
        )

    # Clear the Pinecone semantic cache.
    try:
        if cache_index is not None:
            cache_index.delete(delete_all=True, namespace="")
            print("Semantic cache cleared.")
        else:
            print(
                "Semantic cache unavailable. "
                "Skipping semantic cache clear."
            )

    except Exception as e:
        if "Namespace not found" in str(e):
            print("Semantic cache already empty.")
        else:
            print(
                f"Error clearing semantic cache"
                f"(clear_cache[cache.py]): \n{e}"
            )
            raise


def main():
    redis_client = connect_to_redis()
    cache_index = connect_to_semantic_cache()

    clear_cache(redis_client, cache_index)

    query = "What is Article 21?"

    response = check_semantic_cache(
        cache_index,
        query
    )

    save_to_cache(
        redis_client,
        cache_index,
        query,
        response
    )

    print("\nChecking cache:")
    print(
        check_cache(
            redis_client,
            cache_index,
            query
        )
    )


if __name__ == "__main__":
    main()