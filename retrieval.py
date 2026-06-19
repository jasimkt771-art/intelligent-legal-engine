from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
import cohere
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, COHERE_API_KEY

def initialize_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bm25 = BM25Encoder().load("bm25.json")
    return model, bm25

def connect_to_pinecone():
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

def connect_to_cohere():
    co = cohere.Client(COHERE_API_KEY)
    return co

def generate_query_vectors(query):
    model, bm25 = initialize_models()

    dense_vector = model.encode(query)
    sparse_vector = bm25.encode_queries(query)

    return dense_vector.tolist(), sparse_vector

def hybrid_search(dense_vector, sparse_vector):
    index = connect_to_pinecone()

    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=20,
        include_metadata=True
    )

    return results["matches"]

def rerank_results(query, matches):
    co = connect_to_cohere()

    documents = []

    for match in matches:
        documents.append(match["metadata"]["text"])

    reranked = co.rerank(
        query=query,
        documents=documents,
        top_n=3,
        model="rerank-english-v3.0"
    )

    top_contexts = []

    for result in reranked.results:
        top_contexts.append(documents[result.index])

    return top_contexts

def get_context(query):
    dense_vector, sparse_vector = generate_query_vectors(query)

    matches = hybrid_search(dense_vector, sparse_vector)

    contexts = rerank_results(query, matches)

    return contexts

def main():
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

if __name__ == "__main__":
    main()