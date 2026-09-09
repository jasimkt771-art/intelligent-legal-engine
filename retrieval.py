from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
import cohere

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME1, COHERE_API_KEY


def initialize_models():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            bm25 = BM25Encoder().load("bm25.json")
        except FileNotFoundError as e:
            print(f"BM25 model file not found: \n{e}")
            raise

        return model, bm25

    except Exception as e:
        print(f"Error initializing models(initialize_models[retrieval.py]): \n{e}")
        raise


def connect_to_pinecone():
    try:
        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        index = pc.Index(PINECONE_INDEX_NAME1)

        return index

    except Exception as e:
        print(f"Error connecting to Pinecone(connect_to_pinecone[retrieval.py]): \n{e}")
        raise


def connect_to_cohere():
    try:
        co = cohere.Client(COHERE_API_KEY)

        return co

    except Exception as e:
        print(f"Error connecting to Cohere(connect_to_cohere[retrieval.py]): \n{e}")
        raise


def generate_query_vectors(query):

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    try:
        model, bm25 = initialize_models()

        dense_vector = model.encode(query)
        sparse_vector = bm25.encode_queries(query)

        return dense_vector.tolist(), sparse_vector

    except TypeError as e:
        print(f"Invalid input for vector generation(generate_query_vectors[retrieval.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error generating query vectors(generate_query_vectors[retrieval.py]): \n{e}")
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
            print(f"Missing 'matches' field in Pinecone response(hybrid_search[retrieval.py]): \n{e}")
            raise

        return matches

    except TypeError as e:
        print(f"Invalid vector input for Pinecone search(hybrid_search[retrieval.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error performing hybrid search(hybrid_search[retrieval.py]): \n{e}")
        raise


"""
Format of matches object:
results
└── "matches"
    └── list
        ├── match[0]
        │   ├── "id"
        │   ├── "score"
        │   └── "metadata"
        │       └── "text"
        │
        ├── match[1]
        │   ├── "id"
        │   ├── "score"
        │   └── "metadata"
        │       └── "text"
        │
        └── ...
            ├── "id"
            ├── "score"
            └── "metadata"
                └── "text"
"""


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
        print(f"Missing required match field(rerank_results[retrieval.py]): \n{e}")
        raise

    except TypeError as e:
        print(f"Invalid input for reranking(rerank_results[retrieval.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error reranking results(rerank_results[retrieval.py]): \n{e}")
        raise


def get_context(query):
    try:
        dense_vector, sparse_vector = generate_query_vectors(query)

        matches = hybrid_search(dense_vector, sparse_vector)

        contexts = rerank_results(query, matches)

        return contexts

    except Exception as e:
        print(f"Error getting context(get_context[retrieval.py]): \n{e}")
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
                print(context[:500])
                print("-" * 50)

    except Exception as e:
        print(f"Ingestion failed: \n{e}")


if __name__ == "__main__":
    main()