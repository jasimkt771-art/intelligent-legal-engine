from unittest.mock import patch, call

from app import load_chat_history


# load_chat_history() tests
# Total no of tests: 4


# Display user and assistant messages from chat history
def test_check_displays_user_and_assistant_messages():
    session_id = "3555"
    history = [
        {"user_query": "Article 34", "bot_response": "Article 34 constitutes of ..."}
    ]

    with (
        patch(
            "app.fetch_session_messages", return_value=history
        ) as mock_fetch_session_messages,
        patch("app.st.chat_message") as mock_chat_message,
        patch("app.st.write") as mock_write,
    ):
        load_chat_history(session_id)

        # Verify that chat history was fetched using the correct session ID
        mock_fetch_session_messages.assert_called_once_with(session_id)

        # Verify that both user and assistant chat messages were created
        assert mock_chat_message.call_args_list == [call("user"), call("assistant")]

        # Verify that both messages were displayed in the correct order
        mock_write.assert_has_calls(
            [call("Article 34"), call("Article 34 constitutes of ...")]
        )

        # Verify the exact number of calls
        assert mock_chat_message.call_count == 2
        assert mock_write.call_count == 2


# Empty history displays no chat messages test
def test_check_empty_history_displays_no_chat_messages():
    session_id = "3555"
    history = []

    with (
        patch(
            "app.fetch_session_messages", return_value=history
        ) as mock_fetch_session_messages,
        patch("app.st.chat_message") as mock_chat_message,
        patch("app.st.write") as mock_write,
    ):
        load_chat_history(session_id)

        mock_fetch_session_messages.assert_called_once_with(session_id)
        mock_chat_message.assert_not_called()
        mock_write.assert_not_called()


# render_sidebar() tests


# New Chat button returns the correct result
def test_render_sidebar_new_chat_button_returns_correct_result():
    from app import render_sidebar

    with (
        patch("app.st.button", return_value=True) as mock_button,
        patch("app.get_recent_sessions", return_value=[]) as mock_get_recent_sessions,
    ):
        result = render_sidebar()

        assert result == (True, None)

        mock_button.assert_any_call("+ New Chat")
        mock_get_recent_sessions.assert_called_once_with(limit=5)


# Selecting previous chat returns the correct session ID
def test_render_sidebar_selecting_previous_chat_returns_session_id():
    from app import render_sidebar

    sessions = [{"session_id": "session_123", "first_query": "Article 34"}]

    def mock_button(label, **kwargs):
        if label == "Article 34":
            return True
        return False

    with (
        patch("app.st.button", side_effect=mock_button) as mock_button,
        patch(
            "app.get_recent_sessions", return_value=sessions
        ) as mock_get_recent_sessions,
    ):
        result = render_sidebar()

        assert result == (False, "session_123")

        mock_get_recent_sessions.assert_called_once_with(limit=5)
        mock_button.assert_any_call(
            "Article 34", key="chat_session_123", use_container_width=True
        )
