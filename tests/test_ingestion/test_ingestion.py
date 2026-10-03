from unittest.mock import MagicMock, patch

import ingestion


# Total no of tests: 6


# extract_text_from_pdf() tests

# Text is extracted and combined from all PDF pages
def test_extract_text_from_pdf_combines_page_text():
    first_page = MagicMock()
    first_page.extract_text.return_value = "Article 1\nFirst page."

    second_page = MagicMock()
    second_page.extract_text.return_value = "Article 2\nSecond page."

    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [first_page, second_page]

    with patch(
        "ingestion.PdfReader",
        return_value=mock_pdf_reader,
    ) as mock_reader:
        result = ingestion.extract_text_from_pdf("constitution.pdf")

        assert result == (
            "Article 1\nFirst page."
            "Article 2\nSecond page."
        )

        mock_reader.assert_called_once_with("constitution.pdf")
        first_page.extract_text.assert_called_once_with()
        second_page.extract_text.assert_called_once_with()


# Pages with no extracted text are skipped
def test_extract_text_from_pdf_skips_pages_with_no_text():
    first_page = MagicMock()
    first_page.extract_text.return_value = "Article 1\nFirst page."

    empty_page = MagicMock()
    empty_page.extract_text.return_value = None

    third_page = MagicMock()
    third_page.extract_text.return_value = "Article 3\nThird page."

    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [
        first_page,
        empty_page,
        third_page,
    ]

    with patch(
        "ingestion.PdfReader",
        return_value=mock_pdf_reader,
    ):
        result = ingestion.extract_text_from_pdf("constitution.pdf")

        assert result == (
            "Article 1\nFirst page."
            "Article 3\nThird page."
        )


# extract_articles() tests

# Article headings are detected and text is split into separate articles
def test_extract_articles_splits_text_into_articles():
    text = (
        "Introduction\n"
        "Some introductory text.\n\n"
        "Article 1\n"
        "The first article.\n\n"
        "Article 2A\n"
        "The second article.\n\n"
        "Article 3\n"
        "The third article."
    )

    result = ingestion.extract_articles(text)

    assert result == [
        "Article 1\n"
        "The first article.\n\n",
        "Article 2A\n"
        "The second article.\n\n",
        "Article 3\n"
        "The third article.",
    ]


# Text without Article headings produces no articles
def test_extract_articles_returns_empty_list_when_no_articles_exist():
    text = "This document contains no Article headings."

    result = ingestion.extract_articles(text)

    assert result == []


# generate_hybrid_vectors() tests

# Dense and sparse vectors are generated and stored with the original article text
def test_generate_hybrid_vectors_returns_hybrid_vectors():
    articles = [
        "Article 1\nThe right to equality.",
        "Article 2\nThe right to freedom.",
    ]

    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        MagicMock(
            tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
        ),
        MagicMock(
            tolist=MagicMock(return_value=[0.4, 0.5, 0.6])
        ),
    ]

    mock_bm25 = MagicMock()
    mock_bm25.encode_documents.side_effect = [
        {"indices": [1], "values": [0.8]},
        {"indices": [2], "values": [0.9]},
    ]

    with (
        patch(
            "ingestion.SentenceTransformer",
            return_value=mock_model,
        ) as mock_sentence_transformer,
        patch(
            "ingestion.BM25Encoder",
            return_value=mock_bm25,
        ) as mock_bm25_encoder,
    ):
        result = ingestion.generate_hybrid_vectors(articles)

        assert result == [
            {
                "text": "Article 1\nThe right to equality.",
                "dense_vector": [0.1, 0.2, 0.3],
                "sparse_vector": {
                    "indices": [1],
                    "values": [0.8],
                },
            },
            {
                "text": "Article 2\nThe right to freedom.",
                "dense_vector": [0.4, 0.5, 0.6],
                "sparse_vector": {
                    "indices": [2],
                    "values": [0.9],
                },
            },
        ]

        mock_sentence_transformer.assert_called_once_with(
            "all-MiniLM-L6-v2"
        )
        mock_bm25_encoder.assert_called_once_with()

        mock_bm25.fit.assert_called_once_with(articles)
        mock_bm25.dump.assert_called_once_with("bm25.json")

        assert mock_model.encode.call_count == 2
        assert mock_bm25.encode_documents.call_count == 2


# upsert_to_pinecone() tests

# Hybrid vectors are converted into Pinecone records and uploaded
def test_upsert_to_pinecone_uploads_hybrid_vectors():
    hybrid_vectors = [
        {
            "text": "Article 1\nThe right to equality.",
            "dense_vector": [0.1, 0.2, 0.3],
            "sparse_vector": {
                "indices": [1],
                "values": [0.8],
            },
        },
        {
            "text": "Article 2\nThe right to freedom.",
            "dense_vector": [0.4, 0.5, 0.6],
            "sparse_vector": {
                "indices": [2],
                "values": [0.9],
            },
        },
    ]

    mock_index = MagicMock()
    mock_pinecone = MagicMock()
    mock_pinecone.Index.return_value = mock_index

    generated_ids = [
        "article-id-1",
        "article-id-2",
    ]

    with (
        patch(
            "ingestion.Pinecone",
            return_value=mock_pinecone,
        ) as mock_pinecone_class,
        patch(
            "ingestion.uuid.uuid4",
            side_effect=generated_ids,
        ),
    ):
        result = ingestion.upsert_to_pinecone(hybrid_vectors)

        assert result is True

        mock_pinecone_class.assert_called_once_with(
            api_key=ingestion.PINECONE_API_KEY
        )
        mock_pinecone.Index.assert_called_once_with(
            ingestion.PINECONE_INDEX_NAME1
        )

        mock_index.upsert.assert_called_once_with(
            vectors=[
                {
                    "id": "article-id-1",
                    "values": [0.1, 0.2, 0.3],
                    "sparse_values": {
                        "indices": [1],
                        "values": [0.8],
                    },
                    "metadata": {
                        "text": "Article 1\nThe right to equality."
                    },
                },
                {
                    "id": "article-id-2",
                    "values": [0.4, 0.5, 0.6],
                    "sparse_values": {
                        "indices": [2],
                        "values": [0.9],
                    },
                    "metadata": {
                        "text": "Article 2\nThe right to freedom."
                    },
                },
            ]
        )