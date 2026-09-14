from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
import cohere
import cache

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME1, COHERE_API_KEY


# Load models once when this module is loaded
# Reusable resources
model = cache.embedding_model
bm25 = None
pinecone_index = None
cohere_client = None


def initialize_models():
    global model, bm25

    try:
        # Load the embedding model only once
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load the BM25 encoder only once
        if bm25 is None:
            bm25 = BM25Encoder()
            bm25.load("bm25.json")

        return model, bm25

    except Exception as e:
        print(f"Error initializing retrieval models(initialize_models[retrieval.py]):\n{e}")
        raise


def connect_to_pinecone():
    global pinecone_index

    try:
        if pinecone_index is None:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME1)

        return pinecone_index

    except Exception as e:
        print(f"Error connecting to Pinecone(connect_to_pinecone[retrieval.py]):\n{e}")
        raise

def connect_to_cohere():
    global cohere_client

    try:
        if cohere_client is None:
            cohere_client = cohere.ClientV2(
                api_key=COHERE_API_KEY
            )

        return cohere_client

    except Exception as e:
        print(f"Error connecting to Cohere: {e}")
        raise

def generate_query_vectors(query):

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    try:
        model, bm25 = initialize_models()

        if model is None or bm25 is None:
            raise RuntimeError("Retrieval models could not be initialized.")

        dense_vector = model.encode(query)
        sparse_vector = bm25.encode_queries(query)

        return dense_vector.tolist(), sparse_vector

    except TypeError as e:
        print(
            f"Invalid input for vector generation"
            f"(generate_query_vectors[retrieval.py]): \n{e}"
        )
        raise

    except Exception as e:
        print(
            f"Error generating query vectors"
            f"(generate_query_vectors[retrieval.py]): \n{e}"
        )
        raise


def hybrid_search(dense_vector, sparse_vector):
    try:
        index = connect_to_pinecone()

        results = index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=20,
            include_metadata=True
        )

        try:
            matches = results["matches"]
        except KeyError as e:
            print(
                f"Missing 'matches' field in Pinecone response"
                f"(hybrid_search[retrieval.py]): \n{e}"
            )
            raise

        return matches

    except TypeError as e:
        print(
            f"Invalid vector input for Pinecone search"
            f"(hybrid_search[retrieval.py]): \n{e}"
        )
        raise

    except Exception as e:
        print(
            f"Error performing hybrid search"
            f"(hybrid_search[retrieval.py]): \n{e}"
        )
        raise


def rerank_results(query, matches):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not isinstance(matches, list):
            raise TypeError("Matches must be a list.")

        co = connect_to_cohere()

        documents = []

        for match in matches:
            documents.append(match["metadata"]["text"])

        reranked = co.rerank(
            query=query,
            documents=documents,
            top_n=5,
            model="rerank-v3.5"
        )

        top_contexts = []

        for result in reranked.results:
            top_contexts.append(documents[result.index])

        return top_contexts

    except KeyError as e:
        print(
            f"Missing required match field"
            f"(rerank_results[retrieval.py]): \n{e}"
        )
        raise

    except TypeError as e:
        print(
            f"Invalid input for reranking"
            f"(rerank_results[retrieval.py]): \n{e}"
        )
        raise

    except Exception as e:
        print(
            f"Error reranking results"
            f"(rerank_results[retrieval.py]): \n{e}"
        )
        raise


def get_context(query):
    try:
        dense_vector, sparse_vector = generate_query_vectors(query)

        matches = hybrid_search(dense_vector, sparse_vector)

        contexts = rerank_results(query, matches)

        return contexts

    except Exception as e:
        print(
            f"Error getting context"
            f"(get_context[retrieval.py]): \n{e}"
        )
        raise


def main():
    try:
        while True:
            query = input("\nEnter your query (or type 'exit'): ")

            if query.lower() == "exit":
                print("Goodbye!")
                break

            contexts = get_context(query)

            print(f"\nReturned {len(contexts)} contexts.\n")

            for i, context in enumerate(contexts, start=1):
                print(f"Context {i}:")
                print(context[:100])
                print("-" * 50)

    except Exception as e:
        print(f"Ingestion failed: \n{e}")


if __name__ == "__main__":
    main()