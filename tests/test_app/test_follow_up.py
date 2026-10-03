from app import is_follow_up_query, build_retrieval_query

# Total no of tests: 12

# is_follow_up_query() tests
def test_follow_up_requires_history():
    assert is_follow_up_query("What does that mean?", False) is False


def test_obvious_follow_up_with_history():
    assert is_follow_up_query("What does that mean?", True) is True


def test_reference_follow_up_with_history():
    assert is_follow_up_query("What about that?", True) is True


def test_clarification_follow_up_with_history():
    assert is_follow_up_query("Can you explain that?", True) is True


def test_standalone_question_is_not_follow_up():
    assert is_follow_up_query("What is Article 14?", True) is False


def test_standalone_comparison_is_not_follow_up():
    assert (
        is_follow_up_query(
            "What is the difference between Articles 14 and 15?",
            True,
        )
        is False
    )


def test_case_and_whitespace_are_normalized():
    assert is_follow_up_query("   WHAT DOES THAT MEAN?   ", True) is True


# 2. build_retrieval_query() tests
def test_build_retrieval_query_returns_query_when_history_is_empty():
    result = build_retrieval_query("What is Article 14?", "")

    assert result == "What is Article 14?"


def test_build_retrieval_query_includes_latest_user_and_assistant_messages():
    history = """User: What is Article 14?
Assistant: Article 14 guarantees equality before the law.
User: What about Article 15?
Assistant: Article 15 prohibits certain forms of discrimination."""

    result = build_retrieval_query("Explain that further.", history)

    assert "Previous User Question:\nWhat about Article 15?" in result
    assert (
        "Previous Assistant Answer:\nArticle 15 prohibits certain forms of discrimination."
        in result
    )
    assert "Current User Question:\nExplain that further." in result


def test_build_retrieval_query_uses_latest_messages_from_longer_history():
    history = """User: What is Article 14?
Assistant: It guarantees equality before the law.
User: What is Article 15?
Assistant: It prohibits certain forms of discrimination.
User: What is Article 16?
Assistant: It concerns equality of opportunity in public employment."""

    result = build_retrieval_query("Explain it.", history)

    assert "What is Article 16?" in result
    assert "It concerns equality of opportunity in public employment." in result
    assert "What is Article 14?" not in result
    assert "It guarantees equality before the law." not in result


def test_build_retrieval_query_rejects_non_string_query():
    import pytest

    with pytest.raises(TypeError, match="Query must be a string."):
        build_retrieval_query(None, "")


def test_build_retrieval_query_rejects_non_string_history():
    import pytest

    with pytest.raises(TypeError, match="History must be a string."):
        build_retrieval_query("What is Article 14?", None)
