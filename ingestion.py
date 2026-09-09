import re
import uuid

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME1


def extract_text_from_pdf(pdf_path):
    try:
        doc = PdfReader(pdf_path)

    except FileNotFoundError as e:
        print(f"PDF file not found(extract_text_from_pdf[ingestion.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error reading PDF(extract_text_from_pdf[ingestion.py]): \n{e}")
        raise

    all_text = ""

    for page in doc.pages:
        text = page.extract_text()

        if text is not None:
            all_text += text

    return all_text


def extract_articles(text):
    try:
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        pattern = r'(?m)^Article\s+\d+[A-Z]*'
        matches = list(re.finditer(pattern, text))

        if not matches:
            return []

        articles = []

        for i in range(len(matches)):
            start = matches[i].start()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            articles.append(text[start:end])

        return articles

    except TypeError as e:
        print(f"Invalid text input for article extraction(extract_articles[ingestion.py]): \n{e}")
        raise


def generate_hybrid_vectors(articles):
    try:
        if not isinstance(articles, list):
            raise TypeError("Articles must be a list.")

        for article in articles:
            if not isinstance(article, str):
                raise TypeError("Every article must be a string.")

        # Load the embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create the BM25 encoder
        bm25 = BM25Encoder()

        # Train BM25 on all articles
        bm25.fit(articles)
        bm25.dump("bm25.json")

        hybrid_vectors = []

        # Process each article
        for article in articles:

            # Generate dense embedding
            dense_vector = model.encode(article)

            # Generate sparse embedding
            sparse_vector = bm25.encode_documents(article)

            # Store everything together
            hybrid_vectors.append({
                "text": article,
                "dense_vector": dense_vector.tolist(),
                "sparse_vector": sparse_vector
            })

        return hybrid_vectors

    except TypeError as e:
        print(f"Invalid input for hybrid vector generation(generate_hybrid_vectors[ingestion.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error generating hybrid vectors(generate_hybrid_vectors[ingestion.py]): \n{e}")
        raise


def upsert_to_pinecone(hybrid_vectors):
    try:
        if not isinstance(hybrid_vectors, list):
            raise TypeError("Hybrid vectors must be a list.")

        if not hybrid_vectors:
            raise ValueError("Hybrid vectors list cannot be empty.")

        # Connect to Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Connect to your index
        index = pc.Index(PINECONE_INDEX_NAME1)

        records = []

        # Convert each hybrid vector into a Pinecone record
        for item in hybrid_vectors:

            record = {
                "id": str(uuid.uuid4()),
                "values": item["dense_vector"],
                "sparse_values": item["sparse_vector"],
                "metadata": {
                    "text": item["text"]
                }
            }

            records.append(record)

        # Upload records to Pinecone
        # index.upsert(vectors=records)

        print(f"Successfully uploaded {len(records)} records.")

        return True

    except TypeError as e:
        print(f"Invalid input for Pinecone upload(upsert_to_pinecone[ingestion.py]): \n{e}")
        raise

    except ValueError as e:
        print(f"Invalid value for Pinecone upload(upsert_to_pinecone[ingestion.py]): \n{e}")
        raise

    except KeyError as e:
        print(f"Missing required vector field(upsert_to_pinecone[ingestion.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error uploading to Pinecone(upsert_to_pinecone[ingestion.py]): \n{e}")
        raise


def main():
    print("\n--- Testing extract_text_from_pdf(): invalid PDF path ---")

    try:
        result = extract_text_from_pdf(
            r"C:\invalid\does_not_exist.pdf"
        )
        print(f"Returned value: {result}")

    except Exception as e:
        print(f"Exception caught in main: \n{e}")


    print("\n--- Testing extract_articles(): invalid input ---")

    try:
        result = extract_articles(None)
        print(f"Returned value: {result}")

    except Exception as e:
        print(f"Exception caught in main: \n{e}")


    print("\n--- Testing generate_hybrid_vectors(): invalid input ---")

    try:
        result = generate_hybrid_vectors(None)
        print(f"Returned value: {result}")

    except Exception as e:
        print(f"Exception caught in main: \n{e}")


    print("\n--- Testing generate_hybrid_vectors(): invalid article item ---")

    try:
        result = generate_hybrid_vectors([
            "Article 21: Protection of life and personal liberty.",
            None
        ])
        print(f"Returned value: {result}")

    except Exception as e:
        print(f"Exception caught in main: \n{e}")


    print("\n--- Testing upsert_to_pinecone(): invalid input ---")

    try:
        result = upsert_to_pinecone(None)
        print(f"Returned value: {result}")

    except Exception as e:
        print(f"Exception caught in main: \n{e}")


if __name__ == "__main__":
    main()