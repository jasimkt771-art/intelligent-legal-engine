from unittest.mock import MagicMock, patch

import memory


# Total no of tests: 8


# connect_to_supabase() tests

# Supabase client is created with the configured URL and key
def test_connect_to_supabase_creates_client():
    mock_client = MagicMock()

    with patch(
        "memory.create_client",
        return_value=mock_client,
    ) as mock_create_client:
        result = memory.connect_to_supabase()

        assert result is mock_client

        mock_create_client.assert_called_once_with(
            memory.SUPABASE_URL,
            memory.SUPABASE_KEY,
        )


# fetch_history() tests

# The five most recent messages are requested and returned
def test_fetch_history_returns_recent_messages():
    history = [
        {
            "session_id": "session_1",
            "user_query": "What is Article 21?",
            "bot_response": "Article 21 protects life.",
        },
        {
            "session_id": "session_1",
            "user_query": "Explain Article 19.",
            "bot_response": "Article 19 provides freedoms.",
        },
    ]

    mock_response = MagicMock()
    mock_response.data = history

    mock_query = MagicMock()
    mock_query.table.return_value = mock_query
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query

    with patch(
        "memory.connect_to_supabase",
        return_value=mock_client,
    ):
        result = memory.fetch_history("session_1")

        assert result == history

        mock_client.table.assert_called_once_with("chat_history")
        mock_query.select.assert_called_once_with("*")
        mock_query.eq.assert_called_once_with(
            "session_id",
            "session_1",
        )
        mock_query.order.assert_called_once_with(
            "created_at",
            desc=True,
        )
        mock_query.limit.assert_called_once_with(5)
        mock_query.execute.assert_called_once_with()


# No history data is represented as an empty list
def test_fetch_history_returns_empty_list_when_no_data():
    mock_response = MagicMock()
    mock_response.data = None

    mock_query = MagicMock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query

    with patch(
        "memory.connect_to_supabase",
        return_value=mock_client,
    ):
        result = memory.fetch_history("session_1")

        assert result == []

        mock_client.table.assert_called_once_with("chat_history")
        mock_query.select.assert_called_once_with("*")
        mock_query.eq.assert_called_once_with(
            "session_id",
            "session_1",
        )
        mock_query.order.assert_called_once_with(
            "created_at",
            desc=True,
        )
        mock_query.limit.assert_called_once_with(5)
        mock_query.execute.assert_called_once_with()


# get_recent_sessions() tests

# Conversation rows are grouped into sessions and limited to recent sessions
def test_get_recent_sessions_groups_and_limits_sessions():
    rows = [
        {
            "session_id": "session_2",
            "user_query": "Second session latest query",
            "created_at": "2026-09-30T12:00:00",
        },
        {
            "session_id": "session_1",
            "user_query": "First session latest query",
            "created_at": "2026-09-30T11:00:00",
        },
        {
            "session_id": "session_2",
            "user_query": "Second session first query",
            "created_at": "2026-09-30T10:00:00",
        },
        {
            "session_id": "session_3",
            "user_query": "Third session query",
            "created_at": "2026-09-30T09:00:00",
        },
    ]

    mock_response = MagicMock()
    mock_response.data = rows

    mock_query = MagicMock()
    mock_query.table.return_value = mock_query
    mock_query.select.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query

    with patch(
        "memory.connect_to_supabase",
        return_value=mock_client,
    ):
        result = memory.get_recent_sessions(limit=2)

        assert result == [
            {
                "session_id": "session_2",
                "first_query": "Second session first query",
                "latest_activity": "2026-09-30T12:00:00",
            },
            {
                "session_id": "session_1",
                "first_query": "First session latest query",
                "latest_activity": "2026-09-30T11:00:00",
            },
        ]

        mock_client.table.assert_called_once_with("chat_history")
        mock_query.select.assert_called_once_with(
            "session_id, user_query, created_at"
        )
        mock_query.order.assert_called_once_with(
            "created_at",
            desc=True,
        )
        mock_query.execute.assert_called_once_with()


# fetch_session_messages() tests

# All messages for a session are returned in chronological order
def test_fetch_session_messages_returns_chronological_messages():
    messages = [
        {
            "session_id": "session_1",
            "user_query": "First question",
            "bot_response": "First answer",
            "created_at": "2026-09-30T10:00:00",
        },
        {
            "session_id": "session_1",
            "user_query": "Second question",
            "bot_response": "Second answer",
            "created_at": "2026-09-30T11:00:00",
        },
    ]

    mock_response = MagicMock()
    mock_response.data = messages

    mock_query = MagicMock()
    mock_query.table.return_value = mock_query
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query

    with patch(
        "memory.connect_to_supabase",
        return_value=mock_client,
    ):
        result = memory.fetch_session_messages("session_1")

        assert result == messages

        mock_client.table.assert_called_once_with("chat_history")
        mock_query.select.assert_called_once_with("*")
        mock_query.eq.assert_called_once_with(
            "session_id",
            "session_1",
        )
        mock_query.order.assert_called_once_with(
            "created_at",
            desc=False,
        )
        mock_query.execute.assert_called_once_with()


# save_turn() tests

# A user query and bot response are inserted as one conversation turn
def test_save_turn_inserts_conversation_turn():
    mock_query = MagicMock()

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query
    mock_query.insert.return_value = mock_query

    with patch(
        "memory.connect_to_supabase",
        return_value=mock_client,
    ):
        result = memory.save_turn(
            "session_1",
            "What is Article 21?",
            "Article 21 protects the right to life.",
        )

        assert result is None

        mock_client.table.assert_called_once_with("chat_history")
        mock_query.insert.assert_called_once_with(
            {
                "session_id": "session_1",
                "user_query": "What is Article 21?",
                "bot_response": "Article 21 protects the right to life.",
            }
        )
        mock_query.execute.assert_called_once_with()


# format_history() tests

# Newest-first database history is reversed into chronological LLM context
def test_format_history_reverses_history_and_formats_turns():
    history = [
        {
            "user_query": "Second question",
            "bot_response": "Second answer",
        },
        {
            "user_query": "First question",
            "bot_response": "First answer",
        },
    ]

    result = memory.format_history(history)

    assert result == (
        "User: First question\n"
        "Assistant: First answer\n\n"
        "User: Second question\n"
        "Assistant: Second answer\n\n"
    )


# get_memory() tests

# Retrieved history is formatted and returned as memory context
def test_get_memory_fetches_and_formats_history():
    history = [
        {
            "user_query": "What is Article 21?",
            "bot_response": "Article 21 protects the right to life.",
        }
    ]

    formatted_history = (
        "User: What is Article 21?\n"
        "Assistant: Article 21 protects the right to life.\n\n"
    )

    with (
        patch(
            "memory.fetch_history",
            return_value=history,
        ) as mock_fetch_history,
        patch(
            "memory.format_history",
            return_value=formatted_history,
        ) as mock_format_history,
    ):
        result = memory.get_memory("session_1")

        assert result == formatted_history

        mock_fetch_history.assert_called_once_with("session_1")
        mock_format_history.assert_called_once_with(history)