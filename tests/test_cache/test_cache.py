from unittest.mock import MagicMock, patch

import cache


# Total no of tests: 19


# connect_to_redis() tests

# Existing Redis connection is reused
def test_connect_to_redis_reuses_existing_connection():
    existing_client = MagicMock()

    with patch("cache.redis_client", existing_client):
        result = cache.connect_to_redis()

        assert result is existing_client


# Missing Redis connection creates a configured Redis client
def test_connect_to_redis_creates_connection():
    new_client = MagicMock()

    with (
        patch("cache.redis_client", None),
        patch(
            "cache.redis.Redis",
            return_value=new_client,
        ) as mock_redis,
    ):
        result = cache.connect_to_redis()

        assert result is new_client

        mock_redis.assert_called_once_with(
            host=cache.REDIS_LOCAL_HOST,
            port=cache.REDIS_PORT,
            db=cache.REDIS_DB,
            username=cache.REDIS_USERNAME,
            password=cache.REDIS_PASSWORD,
            decode_responses=True,
        )


# connect_to_semantic_cache() tests

# Existing semantic cache connection is reused
def test_connect_to_semantic_cache_reuses_existing_connection():
    existing_index = MagicMock()

    with patch("cache.semantic_cache_index", existing_index):
        result = cache.connect_to_semantic_cache()

        assert result is existing_index


# Missing semantic cache connection creates the configured Pinecone index
def test_connect_to_semantic_cache_creates_connection():
    mock_index = MagicMock()
    mock_pinecone = MagicMock()
    mock_pinecone.Index.return_value = mock_index

    with (
        patch("cache.semantic_cache_index", None),
        patch(
            "cache.Pinecone",
            return_value=mock_pinecone,
        ) as mock_pinecone_class,
    ):
        result = cache.connect_to_semantic_cache()

        assert result is mock_index

        mock_pinecone_class.assert_called_once_with(
            api_key=cache.PINECONE_API_KEY
        )
        mock_pinecone.Index.assert_called_once_with(
            cache.PINECONE_INDEX_NAME2
        )


# connect_to_cache() tests

# Both Redis and semantic cache connections are returned
def test_connect_to_cache_returns_both_connections():
    mock_redis = MagicMock()
    mock_semantic = MagicMock()

    with (
        patch(
            "cache.connect_to_redis",
            return_value=mock_redis,
        ) as mock_connect_redis,
        patch(
            "cache.connect_to_semantic_cache",
            return_value=mock_semantic,
        ) as mock_connect_semantic,
    ):
        result = cache.connect_to_cache()

        assert result == (mock_redis, mock_semantic)

        mock_connect_redis.assert_called_once_with()
        mock_connect_semantic.assert_called_once_with()


# embed_query() tests

# A query is embedded and converted into a regular list
def test_embed_query_returns_embedding_list():
    mock_model = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.encode.return_value = mock_embedding

    with (
        patch("cache.embedding_model", None),
        patch(
            "cache.SentenceTransformer",
            return_value=mock_model,
        ) as mock_sentence_transformer,
    ):
        result = cache.embed_query("What is Article 21?")

        assert result == [0.1, 0.2, 0.3]

        mock_sentence_transformer.assert_called_once_with(
            "all-MiniLM-L6-v2"
        )
        mock_model.encode.assert_called_once_with(
            "What is Article 21?"
        )
        mock_embedding.tolist.assert_called_once_with()


# Existing embedding model is reused
def test_embed_query_reuses_existing_embedding_model():
    mock_model = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.4, 0.5, 0.6]
    mock_model.encode.return_value = mock_embedding

    with (
        patch("cache.embedding_model", mock_model),
        patch("cache.SentenceTransformer") as mock_sentence_transformer,
    ):
        result = cache.embed_query("What is Article 21?")

        assert result == [0.4, 0.5, 0.6]

        mock_sentence_transformer.assert_not_called()
        mock_model.encode.assert_called_once_with(
            "What is Article 21?"
        )


# check_exact_cache() tests

# Redis exact-cache lookup returns the cached response
def test_check_exact_cache_returns_cached_response():
    mock_redis = MagicMock()
    mock_redis.get.return_value = "Cached response"

    result = cache.check_exact_cache(
        mock_redis,
        "What is Article 21?",
    )

    assert result == "Cached response"

    mock_redis.get.assert_called_once_with(
        "What is Article 21?"
    )


# Unavailable Redis causes exact-cache lookup to be skipped
def test_check_exact_cache_returns_none_when_redis_unavailable():
    result = cache.check_exact_cache(
        None,
        "What is Article 21?",
    )

    assert result is None


# extract_article_number() tests

# Article numbers are extracted case-insensitively
def test_extract_article_number_returns_article_number():
    assert (
        cache.extract_article_number("Explain ARTICLE 21A.")
        == "21a"
    )


# Queries without an Article reference return no article number
def test_extract_article_number_returns_none_without_article():
    assert (
        cache.extract_article_number(
            "What is the right to equality?"
        )
        is None
    )


# check_semantic_cache() tests

# A sufficiently similar cached query returns its cached response
def test_check_semantic_cache_returns_matching_response():
    mock_cache_index = MagicMock()

    mock_match = MagicMock()
    mock_match.score = cache.SEMANTIC_CACHE_THRESHOLD
    mock_match.metadata = {
        "query": "Explain Article 21",
        "response": "Article 21 protects life and personal liberty.",
    }

    mock_results = MagicMock()
    mock_results.matches = [mock_match]
    mock_cache_index.query.return_value = mock_results

    with patch(
        "cache.embed_query",
        return_value=[0.1, 0.2, 0.3],
    ) as mock_embed_query:
        result = cache.check_semantic_cache(
            mock_cache_index,
            "What does Article 21 protect?",
        )

        assert result == (
            "Article 21 protects life and personal liberty."
        )

        mock_embed_query.assert_called_once_with(
            "What does Article 21 protect?"
        )
        mock_cache_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=1,
            include_metadata=True,
        )


# Semantically similar queries about different Articles do not reuse the cache
def test_check_semantic_cache_rejects_different_article():
    mock_cache_index = MagicMock()

    mock_match = MagicMock()
    mock_match.score = 0.99
    mock_match.metadata = {
        "query": "Explain Article 19",
        "response": "Article 19 provides several freedoms.",
    }

    mock_results = MagicMock()
    mock_results.matches = [mock_match]
    mock_cache_index.query.return_value = mock_results

    with patch(
        "cache.embed_query",
        return_value=[0.1, 0.2, 0.3],
    ):
        result = cache.check_semantic_cache(
            mock_cache_index,
            "Explain Article 21",
        )

        assert result is None


# save_exact_cache() tests

# A response is saved in Redis with the configured TTL
def test_save_exact_cache_stores_response_with_ttl():
    mock_redis = MagicMock()

    result = cache.save_exact_cache(
        mock_redis,
        "What is Article 21?",
        "Article 21 protects life.",
    )

    assert result is None

    mock_redis.set.assert_called_once_with(
        "What is Article 21?",
        "Article 21 protects life.",
        ex=cache.REDIS_CACHE_TTL,
    )


# save_semantic_cache() tests

# A query embedding and metadata are uploaded to the semantic cache
def test_save_semantic_cache_uploads_query_and_response():
    mock_cache_index = MagicMock()

    with (
        patch(
            "cache.embed_query",
            return_value=[0.1, 0.2, 0.3],
        ) as mock_embed_query,
        patch(
            "cache.uuid.uuid4",
            return_value="cache-id-1",
        ),
    ):
        result = cache.save_semantic_cache(
            mock_cache_index,
            "What is Article 21?",
            "Article 21 protects life.",
        )

        assert result is None

        mock_embed_query.assert_called_once_with(
            "What is Article 21?"
        )

        mock_cache_index.upsert.assert_called_once_with(
            vectors=[
                {
                    "id": "cache-id-1",
                    "values": [0.1, 0.2, 0.3],
                    "metadata": {
                        "query": "What is Article 21?",
                        "response": "Article 21 protects life.",
                    },
                }
            ]
        )


# check_cache() tests

# An exact cache hit is returned without checking the semantic cache
def test_check_cache_returns_exact_cache_hit_first():
    with (
        patch(
            "cache.check_exact_cache",
            return_value="Exact cached response",
        ) as mock_exact_cache,
        patch(
            "cache.check_semantic_cache",
        ) as mock_semantic_cache,
    ):
        result = cache.check_cache(
            MagicMock(),
            MagicMock(),
            "What is Article 21?",
        )

        assert result == "Exact cached response"

        mock_exact_cache.assert_called_once_with(
            mock_exact_cache.call_args.args[0],
            "What is Article 21?",
        )
        mock_semantic_cache.assert_not_called()


# A semantic cache hit is returned when exact cache misses
def test_check_cache_returns_semantic_cache_hit():
    mock_redis = MagicMock()
    mock_cache_index = MagicMock()

    with (
        patch(
            "cache.check_exact_cache",
            return_value=None,
        ) as mock_exact_cache,
        patch(
            "cache.check_semantic_cache",
            return_value="Semantic cached response",
        ) as mock_semantic_cache,
    ):
        result = cache.check_cache(
            mock_redis,
            mock_cache_index,
            "What is Article 21?",
        )

        assert result == "Semantic cached response"

        mock_exact_cache.assert_called_once_with(
            mock_redis,
            "What is Article 21?",
        )
        mock_semantic_cache.assert_called_once_with(
            mock_cache_index,
            "What is Article 21?",
        )


# No cache hit returns None
def test_check_cache_returns_none_when_no_cache_hit():
    mock_redis = MagicMock()
    mock_cache_index = MagicMock()

    with (
        patch(
            "cache.check_exact_cache",
            return_value=None,
        ),
        patch(
            "cache.check_semantic_cache",
            return_value=None,
        ),
    ):
        result = cache.check_cache(
            mock_redis,
            mock_cache_index,
            "What is Article 21?",
        )

        assert result is None


# save_to_cache() tests

# Both exact and semantic cache entries are saved
def test_save_to_cache_saves_to_both_caches():
    mock_redis = MagicMock()
    mock_cache_index = MagicMock()

    with (
        patch(
            "cache.save_exact_cache",
        ) as mock_save_exact,
        patch(
            "cache.save_semantic_cache",
        ) as mock_save_semantic,
    ):
        result = cache.save_to_cache(
            mock_redis,
            mock_cache_index,
            "What is Article 21?",
            "Article 21 protects life.",
        )

        assert result is None

        mock_save_exact.assert_called_once_with(
            mock_redis,
            "What is Article 21?",
            "Article 21 protects life.",
        )
        mock_save_semantic.assert_called_once_with(
            mock_cache_index,
            "What is Article 21?",
            "Article 21 protects life.",
        )


# clear_cache() tests

# Local Redis, Redis Cloud, and semantic cache are cleared
def test_clear_cache_clears_all_available_caches():
    local_redis = MagicMock()
    semantic_index = MagicMock()
    redis_cloud = MagicMock()

    with patch(
        "cache.redis.Redis",
        return_value=redis_cloud,
    ) as mock_redis:
        result = cache.clear_cache(
            local_redis,
            semantic_index,
        )

        assert result is None

        local_redis.flushdb.assert_called_once_with()

        mock_redis.assert_called_once_with(
            host=cache.REDIS_HOST,
            port=cache.REDIS_PORT,
            db=cache.REDIS_DB,
            username=cache.REDIS_USERNAME,
            password=cache.REDIS_PASSWORD,
            decode_responses=True,
        )

        redis_cloud.flushdb.assert_called_once_with()

        semantic_index.delete.assert_called_once_with(
            delete_all=True,
            namespace="",
        )