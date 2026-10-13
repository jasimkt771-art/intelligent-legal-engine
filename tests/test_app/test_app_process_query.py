
import pytest
from unittest.mock import patch

import app


SESSION_ID = "test-session"
QUERY = "What is Article 14?"
CONTEXTS = ["Article 14 provides equal protection under the law."]
ANSWER = "Article 14 guarantees equal protection under the law."


def test_process_query_returns_cached_response():
    with (
        patch("app.get_memory", return_value=""),
        patch("app.check_cache", return_value="Cached answer") as mock_check_cache,
        patch("app.save_turn") as mock_save_turn,
        patch("app.get_context") as mock_get_context,
        patch("app.generate_response") as mock_generate_response,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        result = app.process_query(
            QUERY, SESSION_ID, "redis-client", "cache-index"
        )

    assert result == "Cached answer"
    mock_check_cache.assert_called_once_with(
        "redis-client", "cache-index", QUERY
    )
    mock_save_turn.assert_called_once_with(
        SESSION_ID, QUERY, "Cached answer"
    )
    mock_get_context.assert_not_called()
    mock_generate_response.assert_not_called()
    mock_save_to_cache.assert_not_called()


def test_process_query_generates_and_caches_independent_answer():
    with (
        patch("app.get_memory", return_value=""),
        patch("app.check_cache", return_value=None) as mock_check_cache,
        patch(
            "app.build_retrieval_query", return_value=QUERY
        ) as mock_build_query,
        patch("app.get_context", return_value=CONTEXTS) as mock_get_context,
        patch(
            "app.generate_response",
            return_value={
                "answer": ANSWER,
                "is_follow_up": False,
            },
        ) as mock_generate_response,
        patch("app.save_turn") as mock_save_turn,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        result = app.process_query(
            QUERY, SESSION_ID, "redis-client", "cache-index"
        )

    assert result == ANSWER
    mock_check_cache.assert_called_once_with(
        "redis-client", "cache-index", QUERY
    )
    mock_build_query.assert_called_once_with(QUERY, "")
    mock_get_context.assert_called_once_with(QUERY)
    mock_generate_response.assert_called_once_with(QUERY, CONTEXTS, "")
    mock_save_turn.assert_called_once_with(SESSION_ID, QUERY, ANSWER)
    mock_save_to_cache.assert_called_once_with(
        "redis-client", "cache-index", QUERY, ANSWER
    )


def test_process_query_does_not_cache_llm_classified_follow_up():
    history = (
        "User: What is Article 14?\n"
        "Assistant: Article 14 guarantees equal protection under the law."
    )
    follow_up_query = "Does this protection extend to prisoners?"
    retrieval_query = (
        "Recent Conversation (up to 3 exchanges):\n"
        f"{history}\n\n"
        f"Current User Question:\n{follow_up_query}"
    )
    answer = "The retrieved context does not establish that."

    with (
        patch("app.get_memory", return_value=history),
        patch("app.check_cache", return_value=None) as mock_check_cache,
        patch(
            "app.build_retrieval_query",
            return_value=retrieval_query,
        ) as mock_build_query,
        patch("app.get_context", return_value=CONTEXTS) as mock_get_context,
        patch(
            "app.generate_response",
            return_value={
                "answer": answer,
                "is_follow_up": True,
            },
        ) as mock_generate_response,
        patch("app.save_turn") as mock_save_turn,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        result = app.process_query(
            follow_up_query, SESSION_ID, "redis-client", "cache-index"
        )

    assert result == answer
    mock_check_cache.assert_called_once_with(
        "redis-client", "cache-index", follow_up_query
    )
    mock_build_query.assert_called_once_with(follow_up_query, history)
    mock_get_context.assert_called_once_with(retrieval_query)
    mock_generate_response.assert_called_once_with(
        retrieval_query, CONTEXTS, history
    )
    mock_save_turn.assert_called_once_with(
        SESSION_ID, follow_up_query, answer
    )
    mock_save_to_cache.assert_not_called()


def test_process_query_caches_when_llm_classifies_query_as_independent():
    with (
        patch("app.get_memory", return_value=""),
        patch("app.check_cache", return_value=None),
        patch("app.build_retrieval_query", return_value=QUERY),
        patch("app.get_context", return_value=CONTEXTS),
        patch(
            "app.generate_response",
            return_value={
                "answer": ANSWER,
                "is_follow_up": False,
            },
        ),
        patch("app.save_turn") as mock_save_turn,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        result = app.process_query(
            QUERY, SESSION_ID, "redis-client", "cache-index"
        )

    assert result == ANSWER
    mock_save_turn.assert_called_once_with(SESSION_ID, QUERY, ANSWER)
    mock_save_to_cache.assert_called_once_with(
        "redis-client", "cache-index", QUERY, ANSWER
    )


def test_process_query_propagates_generation_errors():
    with (
        patch("app.get_memory", return_value=""),
        patch("app.check_cache", return_value=None),
        patch("app.build_retrieval_query", return_value=QUERY),
        patch("app.get_context", return_value=CONTEXTS),
        patch(
            "app.generate_response",
            side_effect=ValueError("LLM generation failed"),
        ),
        patch("app.save_turn") as mock_save_turn,
        patch("app.save_to_cache") as mock_save_to_cache,
    ):
        with pytest.raises(ValueError, match="LLM generation failed"):
            app.process_query(
                QUERY, SESSION_ID, "redis-client", "cache-index"
            )

    mock_save_turn.assert_not_called()
    mock_save_to_cache.assert_not_called()


def test_process_query_rejects_invalid_query():
    with pytest.raises(TypeError, match="Query must be a string"):
        app.process_query(
            None, SESSION_ID, "redis-client", "cache-index"
        )
