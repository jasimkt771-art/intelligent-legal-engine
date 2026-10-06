from unittest.mock import patch

from app import process_query


# process_query() tests
# Total no of tests: 3

# Cache hit test
def test_process_query_returns_cached_response_without_generating_new_response():
    query = "What is Article 14?"
    session_id = "session-123"
    redis_client = object()
    cache_index = object()
    cached_response = "Cached legal answer."

    with (
        patch("app.get_memory", return_value="") as mock_get_memory,
        patch("app.check_cache", return_value=cached_response) as mock_check_cache,
        patch("app.get_context") as mock_get_context,
        patch("app.generate_response") as mock_generate_response,
        patch("app.save_turn") as mock_save_turn,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        result = process_query(
            query,
            session_id,
            redis_client,
            cache_index,
        )

    assert result == cached_response

    mock_get_memory.assert_called_once_with(session_id)
    mock_check_cache.assert_called_once_with(
        redis_client,
        cache_index,
        query,
    )

    mock_get_context.assert_not_called()
    mock_generate_response.assert_not_called()
    mock_save_turn.assert_called_once_with(
session_id,
        query,
        cached_response,
    )
    mock_save_to_cache.assert_not_called()


# LLM hit test
def test_process_query_generates_and_caches_response_on_cache_miss():
    query = "What is Article 14?"
    session_id = "session-123"
    redis_client = object()
    cache_index = object()
    contexts = ["Article 14 guarantees equality before the law."]
    generated_response = "Article 14 guarantees equality before the law."

    with (
        patch("app.get_memory", return_value="") as mock_get_memory,
        patch("app.check_cache", return_value=None) as mock_check_cache,
        patch("app.get_context", return_value=contexts) as mock_get_context,
        patch(
            "app.generate_response",
            return_value=generated_response,
        ) as mock_generate_response,
        patch("app.save_turn") as mock_save_turn,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        result = process_query(
            query,
            session_id,
            redis_client,
            cache_index,
        )

    assert result == generated_response

    mock_get_memory.assert_called_once_with(session_id)
    mock_check_cache.assert_called_once_with(
        redis_client,
        cache_index,
        query,
    )
    mock_get_context.assert_called_once_with(query)
    mock_generate_response.assert_called_once_with(
        query,
        contexts,
        "",
    )
    mock_save_turn.assert_called_once_with(
        session_id,
        query,
        generated_response,
    )
    mock_save_to_cache.assert_called_once_with(
        redis_client,
        cache_index,
        query,
        generated_response,
    )


# Follow-up query bypass cache test
def test_process_query_bypasses_cache_for_follow_up():
    query = "What about that?"
    session_id = "session-123"
    redis_client = object()
    cache_index = object()

    history = (
        "User: What is Article 14?\n"
        "Assistant: Article 14 guarantees equality before the law."
    )

    retrieval_query = (
        "Previous User Question:\nWhat is Article 14?\n"
        "Previous Assistant Answer:\n"
        "Article 14 guarantees equality before the law.\n"
        "Current User Question:\nWhat about that?"
    )

    contexts = ["Retrieved legal context."]
    generated_response = "Here is the explanation."

    with (
        patch("app.get_memory", return_value=history) as mock_get_memory,
        patch("app.check_cache") as mock_check_cache,
        patch(
            "app.build_retrieval_query",
            return_value=retrieval_query,
        ) as mock_build_retrieval_query,
        patch("app.get_context", return_value=contexts) as mock_get_context,
        patch(
            "app.generate_response",
            return_value=generated_response,
        ) as mock_generate_response,
        patch("app.save_turn") as mock_save_turn,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        result = process_query(
            query,
            session_id,
            redis_client,
            cache_index,
        )

    assert result == generated_response

    mock_get_memory.assert_called_once_with(session_id)
    mock_check_cache.assert_not_called()

    mock_build_retrieval_query.assert_called_once_with(
        query,
        history,
    )
    mock_get_context.assert_called_once_with(retrieval_query)

    mock_generate_response.assert_called_once_with(
        query,
        contexts,
        history,
    )
    mock_save_turn.assert_called_once_with(
        session_id,
        query,
        generated_response,
    )
    mock_save_to_cache.assert_not_called()
