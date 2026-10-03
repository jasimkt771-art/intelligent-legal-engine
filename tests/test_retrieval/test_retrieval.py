from unittest.mock import MagicMock, patch

import retrieval


# Total no of tests: 10


# initialize_models() tests

# Existing models are reused without creating them again
def test_initialize_models_reuses_existing_models():
    existing_model = MagicMock()
    existing_bm25 = MagicMock()

    with (
        patch("retrieval.model", existing_model),
        patch("retrieval.bm25", existing_bm25),
        patch("retrieval.SentenceTransformer") as mock_sentence_transformer,
        patch("retrieval.BM25Encoder") as mock_bm25_encoder,
    ):
        result = retrieval.initialize_models()

        assert result == (existing_model, existing_bm25)
        mock_sentence_transformer.assert_not_called()
        mock_bm25_encoder.assert_not_called()


# Missing models are initialized and BM25 data is loaded
def test_initialize_models_creates_missing_models():
    new_model = MagicMock()
    new_bm25 = MagicMock()

    with (
        patch("retrieval.model", None),
        patch("retrieval.bm25", None),
        patch(
            "retrieval.SentenceTransformer",
            return_value=new_model,
        ) as mock_sentence_transformer,
        patch(
            "retrieval.BM25Encoder",
            return_value=new_bm25,
        ) as mock_bm25_encoder,
    ):
        result = retrieval.initialize_models()

        assert result == (new_model, new_bm25)

        mock_sentence_transformer.assert_called_once_with(
            "all-MiniLM-L6-v2"
        )
        mock_bm25_encoder.assert_called_once_with()
        new_bm25.load.assert_called_once_with("bm25.json")


# connect_to_pinecone() tests

# Existing Pinecone connection is reused
def test_connect_to_pinecone_reuses_existing_connection():
    existing_index = MagicMock()

    with patch("retrieval.pinecone_index", existing_index):
        result = retrieval.connect_to_pinecone()

        assert result is existing_index


# Missing Pinecone connection creates and returns an index
def test_connect_to_pinecone_creates_connection():
    mock_index = MagicMock()
    mock_pinecone = MagicMock()
    mock_pinecone.Index.return_value = mock_index

    with (
        patch("retrieval.pinecone_index", None),
        patch(
            "retrieval.Pinecone",
            return_value=mock_pinecone,
        ) as mock_pinecone_class,
    ):
        result = retrieval.connect_to_pinecone()

        assert result is mock_index

        mock_pinecone_class.assert_called_once_with(
            api_key=retrieval.PINECONE_API_KEY
        )
        mock_pinecone.Index.assert_called_once_with(
            retrieval.PINECONE_INDEX_NAME1
        )


# connect_to_cohere() tests

# Existing Cohere connection is reused
def test_connect_to_cohere_reuses_existing_connection():
    existing_client = MagicMock()

    with patch("retrieval.cohere_client", existing_client):
        result = retrieval.connect_to_cohere()

        assert result is existing_client


# Missing Cohere connection creates a client
def test_connect_to_cohere_creates_connection():
    new_client = MagicMock()

    with (
        patch("retrieval.cohere_client", None),
        patch("retrieval.cohere") as mock_cohere,
    ):
        mock_cohere.ClientV2.return_value = new_client

        result = retrieval.connect_to_cohere()

        assert result is new_client

        mock_cohere.ClientV2.assert_called_once_with(
            api_key=retrieval.COHERE_API_KEY
        )


# generate_query_vectors() tests

# Query is converted into dense and sparse vectors
def test_generate_query_vectors_returns_dense_and_sparse_vectors():
    mock_model = MagicMock()
    mock_bm25 = MagicMock()

    mock_dense_vector = MagicMock()
    mock_dense_vector.tolist.return_value = [0.1, 0.2, 0.3]

    mock_model.encode.return_value = mock_dense_vector
    mock_bm25.encode_queries.return_value = {
        "indices": [1, 2],
        "values": [0.5, 0.8],
    }

    with patch(
        "retrieval.initialize_models",
        return_value=(mock_model, mock_bm25),
    ):
        dense_vector, sparse_vector = retrieval.generate_query_vectors(
            "What is Article 21?"
        )

        assert dense_vector == [0.1, 0.2, 0.3]
        assert sparse_vector == {
            "indices": [1, 2],
            "values": [0.5, 0.8],
        }

        mock_model.encode.assert_called_once_with("What is Article 21?")
        mock_bm25.encode_queries.assert_called_once_with(
            "What is Article 21?"
        )


# hybrid_search() tests

# Pinecone query returns the retrieved matches
def test_hybrid_search_returns_matches():
    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {"id": "article_21", "score": 0.95},
            {"id": "article_22", "score": 0.88},
        ]
    }

    with patch(
        "retrieval.connect_to_pinecone",
        return_value=mock_index,
    ):
        result = retrieval.hybrid_search(
            [0.1, 0.2, 0.3],
            {"indices": [1], "values": [0.8]},
        )

        assert result == [
            {"id": "article_21", "score": 0.95},
            {"id": "article_22", "score": 0.88},
        ]

        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            sparse_vector={"indices": [1], "values": [0.8]},
            top_k=20,
            include_metadata=True,
        )


# rerank_results() tests

# Retrieved documents are reranked and converted into contexts
def test_rerank_results_returns_top_contexts():
    mock_cohere = MagicMock()

    mock_cohere.rerank.return_value.results = [
        MagicMock(index=1),
        MagicMock(index=0),
    ]

    matches = [
        {
            "metadata": {
                "text": "Article 21 protects life and personal liberty."
            }
        },
        {
            "metadata": {
                "text": "Article 19 provides several fundamental freedoms."
            }
        },
    ]

    with patch(
        "retrieval.connect_to_cohere",
        return_value=mock_cohere,
    ):
        result = retrieval.rerank_results(
            "What does Article 21 protect?",
            matches,
        )

        assert result == [
            "Article 19 provides several fundamental freedoms.",
            "Article 21 protects life and personal liberty.",
        ]

        mock_cohere.rerank.assert_called_once_with(
            query="What does Article 21 protect?",
            documents=[
                "Article 21 protects life and personal liberty.",
                "Article 19 provides several fundamental freedoms.",
            ],
            top_n=5,
            model="rerank-v3.5",
        )


# get_context() tests

# The complete retrieval pipeline passes data through each stage
def test_get_context_runs_complete_retrieval_pipeline():
    dense_vector = [0.1, 0.2, 0.3]
    sparse_vector = {"indices": [1], "values": [0.8]}
    matches = [{"id": "article_21"}]
    contexts = [
        "Article 21 protects life and personal liberty."
    ]

    with (
        patch(
            "retrieval.generate_query_vectors",
            return_value=(dense_vector, sparse_vector),
        ) as mock_generate_query_vectors,
        patch(
            "retrieval.hybrid_search",
            return_value=matches,
        ) as mock_hybrid_search,
        patch(
            "retrieval.rerank_results",
            return_value=contexts,
        ) as mock_rerank_results,
    ):
        result = retrieval.get_context("What is Article 21?")

        assert result == contexts

        mock_generate_query_vectors.assert_called_once_with(
            "What is Article 21?"
        )
        mock_hybrid_search.assert_called_once_with(
            dense_vector,
            sparse_vector,
        )
        mock_rerank_results.assert_called_once_with(
            "What is Article 21?",
            matches,
        )