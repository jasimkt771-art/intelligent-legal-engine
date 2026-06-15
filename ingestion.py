import re
import uuid

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

def extract_text_from_pdf(pdf_path):
    doc = PdfReader(pdf_path)
    all_text = ""

    for page in doc.pages:
        text = page.extract_text()
        if text is not None:
            all_text += text
    
    return all_text        
    
def extract_articles(text): 
    pattern = r'(?m)^Article\s+\d+[A-Z]*' 
    matches = list(re.finditer(pattern, text)) 
    if not matches: 
        return [] 
    
    articles = [] 
    for i in range(len(matches)): 
        start = matches[i].start() 
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text) 
        articles.append(text[start:end]) 
    return articles

def generate_hybrid_vectors(articles):
    try:
        # Load the embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create the BM25 encoder
        bm25 = BM25Encoder()

        # Train BM25 on all articles
        bm25.fit(articles)

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

    except Exception as e:
        print(f"Error generating hybrid vectors: {e}")
        return []

def upsert_to_pinecone(hybrid_vectors):
    try:
        # Connect to Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Connect to your index
        index = pc.Index(PINECONE_INDEX_NAME)

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
        index.upsert(vectors=records)

        print(f"Successfully uploaded {len(records)} records.")

    except Exception as e:
        print(f"Error uploading to Pinecone: {e}")

def main():
    try:
        pdf_path = r"C:\Users\hp\OneDrive\Desktop\Project 6\3. Implementation\Data\Constitution_Edited.pdf"
        text = extract_text_from_pdf(pdf_path)
        articles = extract_articles(text)
        hybrid_vectors = generate_hybrid_vectors(articles)
        upsert_to_pinecone(hybrid_vectors)
        print("Ingestion completed successfully.")

    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()