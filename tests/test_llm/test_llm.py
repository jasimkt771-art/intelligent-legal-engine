from unittest.mock import MagicMock, patch

import llm


# Total no of tests: 10


# build_prompt() tests

# Query, history, and retrieved contexts are included in the generated prompt
def test_build_prompt_includes_query_history_and_contexts():
    result = llm.build_prompt(
        "What is Article 21?",
        [
            "Article 21 protects life and personal liberty.",
            "Article 19 provides several freedoms.",
        ],
        "User: Explain Article 19.\nAssistant: It provides freedoms.",
    )

    assert "What is Article 21?" in result
    assert "User: Explain Article 19." in result
    assert "Assistant: It provides freedoms." in result
    assert "Article 21 protects life and personal liberty." in result
    assert "Article 19 provides several freedoms." in result
    assert "You are a legal assistant." in result
    assert "Retrieved Context is the only source of truth" in result


# Multiple contexts are joined with blank lines
def test_build_prompt_separates_contexts():
    result = llm.build_prompt(
        "What is Article 21?",
        ["First context", "Second context", "Third context"],
        "",
    )

    assert "First context\n\nSecond context\n\nThird context" in result


# generate_gemini_response() tests

# Gemini receives the prompt with Gemini-specific instructions and returns its text
def test_generate_gemini_response_returns_response_text():
    mock_response = MagicMock()
    mock_response.text = "Gemini legal response."

    with patch(
        "llm.client.models.generate_content",
        return_value=mock_response,
    ) as mock_generate:
        result = llm.generate_gemini_response("Base prompt")

        assert result == "Gemini legal response."

        mock_generate.assert_called_once()

        call_kwargs = mock_generate.call_args.kwargs

        assert call_kwargs["model"] == llm.GEMINI_MODEL
        assert "Base prompt" in call_kwargs["contents"]
        assert "Gemini-Specific Instructions:" in call_kwargs["contents"]


# generate_groq_response() tests

# Groq receives the prompt with Groq-specific instructions and returns message content
def test_generate_groq_response_returns_response_content():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Groq legal response."

    with patch(
        "llm.groq_client.chat.completions.create",
        return_value=mock_response,
    ) as mock_create:
        result = llm.generate_groq_response("Base prompt")

        assert result == "Groq legal response."

        mock_create.assert_called_once()

        call_kwargs = mock_create.call_args.kwargs

        assert call_kwargs["model"] == llm.GROQ_MODEL
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert "Base prompt" in call_kwargs["messages"][0]["content"]
        assert "Groq-Specific Instructions:" in call_kwargs["messages"][0]["content"]


# generate_response() tests

# Ollama is used directly when the configured provider is Ollama
def test_generate_response_uses_ollama():
    ollama_response = {
        "message": {
            "content": "Ollama legal response."
        }
    }

    with (
        patch("llm.LLM_PROVIDER", "ollama"),
        patch(
            "llm.build_prompt",
            return_value="Built prompt",
        ) as mock_build_prompt,
        patch(
            "llm.print_prompt_preview",
        ) as mock_preview,
        patch(
            "llm.ollama.chat",
            return_value=ollama_response,
        ) as mock_chat,
    ):
        result = llm.generate_response(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
        )

        assert result == "Ollama legal response."

        mock_build_prompt.assert_called_once_with(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
        )

        mock_preview.assert_called_once_with(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
            max_chars=200,
        )

        call_kwargs = mock_chat.call_args.kwargs

        assert call_kwargs["model"] == llm.OLLAMA_MODEL
        assert call_kwargs["messages"][0]["role"] == "user"
        assert "Built prompt" in call_kwargs["messages"][0]["content"]
        assert "Ollama-Specific Instructions:" in call_kwargs["messages"][0]["content"]


# Groq is used as the primary provider when configured
def test_generate_response_uses_groq():
    with (
        patch("llm.LLM_PROVIDER", llm.GROQ),
        patch(
            "llm.build_prompt",
            return_value="Built prompt",
        ) as mock_build_prompt,
        patch(
            "llm.print_prompt_preview",
        ) as mock_preview,
        patch(
            "llm.generate_groq_response",
            return_value="Groq legal response.",
        ) as mock_groq,
    ):
        result = llm.generate_response(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
        )

        assert result == "Groq legal response."

        mock_build_prompt.assert_called_once_with(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
        )

        mock_preview.assert_called_once_with(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
            max_chars=200,
        )

        mock_groq.assert_called_once_with("Built prompt")


# Gemini is used directly when configured as the provider
def test_generate_response_uses_gemini():
    with (
        patch("llm.LLM_PROVIDER", llm.GEMINI),
        patch(
            "llm.build_prompt",
            return_value="Built prompt",
        ) as mock_build_prompt,
        patch(
            "llm.print_prompt_preview",
        ) as mock_preview,
        patch(
            "llm.generate_gemini_response",
            return_value="Gemini legal response.",
        ) as mock_gemini,
    ):
        result = llm.generate_response(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
        )

        assert result == "Gemini legal response."

        mock_build_prompt.assert_called_once_with(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
        )

        mock_preview.assert_called_once_with(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
            max_chars=200,
        )

        mock_gemini.assert_called_once_with("Built prompt")


# Groq failure triggers Gemini fallback using the same built prompt
def test_generate_response_falls_back_to_gemini_when_groq_fails():
    with (
        patch("llm.LLM_PROVIDER", llm.GROQ),
        patch(
            "llm.build_prompt",
            return_value="Built prompt",
        ),
        patch(
            "llm.print_prompt_preview",
        ),
        patch(
            "llm.generate_groq_response",
            side_effect=RuntimeError("Groq unavailable"),
        ) as mock_groq,
        patch(
            "llm.generate_gemini_response",
            return_value="Gemini fallback response.",
        ) as mock_gemini,
    ):
        result = llm.generate_response(
            "What is Article 21?",
            ["Article 21 context"],
            "Previous history",
        )

        assert result == "Gemini fallback response."

        mock_groq.assert_called_once_with("Built prompt")
        mock_gemini.assert_called_once_with("Built prompt")