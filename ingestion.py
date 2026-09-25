import re
import uuid

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME1


# Offline ingestion pipeline:
# PDF → Articles → Dense + Sparse vectors → Pinecone


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

        # Match Article headings at the beginning of a line.
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
    """
    Generate dense semantic vectors and sparse BM25 vectors
    for each Article.
    """
    try:
        if not isinstance(articles, list):
            raise TypeError("Articles must be a list.")

        for article in articles:
            if not isinstance(article, str):
                raise TypeError("Every article must be a string.")

        model = SentenceTransformer("all-MiniLM-L6-v2")

        bm25 = BM25Encoder()
        bm25.fit(articles)

        # Save the fitted BM25 model for use during online retrieval.
        bm25.dump("bm25.json")

        hybrid_vectors = []

        for article in articles:
            dense_vector = model.encode(article)
            sparse_vector = bm25.encode_documents(article)

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
    """
    Upload hybrid vectors to Pinecone with the original
    Article text stored as metadata.
    """
    try:
        if not isinstance(hybrid_vectors, list):
            raise TypeError("Hybrid vectors must be a list.")

        if not hybrid_vectors:
            raise ValueError("Hybrid vectors list cannot be empty.")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME1)

        records = []

        for item in hybrid_vectors:
            records.append({
                "id": str(uuid.uuid4()),
                "values": item["dense_vector"],
                "sparse_values": item["sparse_vector"],
                "metadata": {
                    "text": item["text"]
                }
            })

        index.upsert(vectors=records)

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
    pdf_path = r"C:\Users\hp\OneDrive\Desktop\Project 6\3. Implementation\Data\Final Constitution2.pdf"

    print("\n--- Testing complete ingestion pipeline ---")

    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted text length: {len(text)} characters")

        articles = extract_articles(text)
        print(f"Extracted {len(articles)} articles")

        hybrid_vectors = generate_hybrid_vectors(articles)
        print(f"Generated {len(hybrid_vectors)} hybrid vectors")

        for i, vector in enumerate(hybrid_vectors, start=1):
            print(f"Article {i}")
            print(f'\nDense Vector:\n{vector["dense_vector"][:5]}')
            print(f'\nSparse Vector:\n{vector["sparse_vector"]}')
            print(vector["text"][:100])

        result = upsert_to_pinecone(hybrid_vectors)
        print(f"Pinecone upload result: {result}")

        print("\n--- Ingestion pipeline completed successfully ---")

    except Exception as e:
        print(f"Error in ingestion pipeline(main[ingestion.py]): \n{e}")


if __name__ == "__main__":
    main()