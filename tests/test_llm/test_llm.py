
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import llm


VALID_RESPONSE = json.dumps(
    {
        "answer": "Article 14 guarantees equality before the law.",
        "is_follow_up": False,
    }
)

VALID_FOLLOW_UP_RESPONSE = json.dumps(
    {
        "answer": "Yes, based on the context provided.",
        "is_follow_up": True,
    }
)


def test_parse_llm_response_returns_validated_dictionary():
    result = llm.parse_llm_response(VALID_RESPONSE)

    assert result == {
        "answer": "Article 14 guarantees equality before the law.",
        "is_follow_up": False,
    }


def test_parse_llm_response_accepts_follow_up_true():
    result = llm.parse_llm_response(VALID_FOLLOW_UP_RESPONSE)

    assert result["is_follow_up"] is True


@pytest.mark.parametrize(
    "raw_response",
    [
        "not JSON",
        '{"answer": "Missing classification"}',
        '{"is_follow_up": false}',
        '{"answer": "Answer", "is_follow_up": false, "extra": 1}',
        '{"answer": "", "is_follow_up": false}',
        '{"answer": "Answer", "is_follow_up": "false"}',
        '{"answer": "Answer", "is_follow_up": null}',
        "[]",
        "null",
    ],
)
def test_parse_llm_response_rejects_invalid_responses(raw_response):
    with pytest.raises((ValueError, TypeError)):
        llm.parse_llm_response(raw_response)


@pytest.mark.parametrize(
    "raw_response",
    [None, 123, {"answer": "Answer"}],
)
def test_parse_llm_response_rejects_non_string_input(raw_response):
    with pytest.raises(TypeError):
        llm.parse_llm_response(raw_response)


def test_build_prompt_includes_query_context_history_and_json_requirements():
    query = "What is Article 14?"
    contexts = ["Article 14 guarantees equality before the law."]
    history = (
        "User: Explain equality.\n"
        "Assistant: Equality means equal treatment."
    )

    prompt = llm.build_prompt(query, contexts, history)

    assert query in prompt
    assert contexts[0] in prompt
    assert history in prompt
    assert '"answer"' in prompt
    assert '"is_follow_up"' in prompt
    assert "JSON object" in prompt


@pytest.mark.parametrize(
    ("query", "contexts", "history"),
    [
        (None, [], ""),
        ("Question?", "not a list", ""),
        ("Question?", [], None),
    ],
)
def test_build_prompt_rejects_invalid_inputs(query, contexts, history):
    with pytest.raises(TypeError):
        llm.build_prompt(query, contexts, history)


def test_generate_ollama_response_requests_json_format():
    mock_response = {
        "message": {
            "content": VALID_RESPONSE,
        }
    }

    with patch(
        "llm.ollama.chat",
        return_value=mock_response,
    ) as mock_chat:
        result = llm.generate_ollama_response("Test prompt")

    assert result == VALID_RESPONSE
    mock_chat.assert_called_once()

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["model"] == llm.OLLAMA_MODEL
    assert kwargs["format"] == "json"
    assert kwargs["messages"][0]["content"].startswith("Test prompt")


def test_generate_ollama_response_raises_for_empty_content():
    mock_response = {"message": {"content": ""}}

    with patch("llm.ollama.chat", return_value=mock_response):
        with pytest.raises(ValueError, match="empty or invalid"):
            llm.generate_ollama_response("Test prompt")


def test_generate_groq_response_requests_json_object():
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=VALID_RESPONSE)
            )
        ]
    )

    with patch(
        "llm.groq_client.chat.completions.create",
        return_value=mock_response,
    ) as mock_create:
        result = llm.generate_groq_response("Test prompt")

    assert result == VALID_RESPONSE

    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == llm.GROQ_MODEL
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0
    assert kwargs["messages"][0]["content"].startswith("Test prompt")


def test_generate_groq_response_propagates_api_errors():
    with patch(
        "llm.groq_client.chat.completions.create",
        side_effect=RuntimeError("Groq unavailable"),
    ):
        with pytest.raises(RuntimeError, match="Groq unavailable"):
            llm.generate_groq_response("Test prompt")


def test_generate_gemini_response_requests_json():
    mock_response = SimpleNamespace(text=VALID_RESPONSE)

    with patch(
        "llm.client.models.generate_content",
        return_value=mock_response,
    ) as mock_generate:
        result = llm.generate_gemini_response("Test prompt")

    assert result == VALID_RESPONSE

    kwargs = mock_generate.call_args.kwargs
    assert kwargs["model"] == llm.GEMINI_MODEL
    assert kwargs["contents"].startswith("Test prompt")
    assert kwargs["config"] == {
        "response_mime_type": "application/json"
    }


def test_generate_gemini_response_raises_when_text_is_missing():
    mock_response = SimpleNamespace(text=None)

    with patch(
        "llm.client.models.generate_content",
        return_value=mock_response,
    ):
        with pytest.raises(ValueError, match="no response text"):
            llm.generate_gemini_response("Test prompt")


def test_generate_response_uses_ollama():
    with (
        patch("llm.LLM_PROVIDER", llm.OLLAMA),
        patch(
            "llm.generate_ollama_response",
            return_value=VALID_RESPONSE,
        ) as mock_ollama,
    ):
        result, attempts = llm.generate_response(
            "What is Article 14?",
            ["Article 14 guarantees equality."],
            "",
        )

    assert result == json.loads(VALID_RESPONSE)
    assert attempts == 1
    mock_ollama.assert_called_once()


def test_generate_response_uses_groq():
    with (
        patch("llm.LLM_PROVIDER", llm.GROQ),
        patch(
            "llm.generate_groq_response",
            return_value=VALID_RESPONSE,
        ) as mock_groq,
        patch("llm.generate_gemini_response") as mock_gemini,
    ):
        result, attempts = llm.generate_response(
            "What is Article 14?",
            ["Article 14 guarantees equality."],
            "",
        )

    assert result == json.loads(VALID_RESPONSE)
    assert attempts == 1
    mock_groq.assert_called_once()
    mock_gemini.assert_not_called()


def test_generate_response_uses_gemini():
    with (
        patch("llm.LLM_PROVIDER", llm.GEMINI),
        patch(
            "llm.generate_gemini_response",
            return_value=VALID_RESPONSE,
        ) as mock_gemini,
        patch("llm.generate_groq_response") as mock_groq,
    ):
        result, attempts = llm.generate_response(
            "What is Article 14?",
            ["Article 14 guarantees equality."],
            "",
        )

    assert result == json.loads(VALID_RESPONSE)
    assert attempts == 1
    mock_gemini.assert_called_once()
    mock_groq.assert_not_called()


def test_generate_response_falls_back_to_gemini_when_groq_fails():
    with (
        patch("llm.LLM_PROVIDER", llm.GROQ),
        patch(
            "llm.generate_groq_response",
            side_effect=RuntimeError("Groq unavailable"),
        ) as mock_groq,
        patch(
            "llm.generate_gemini_response",
            return_value=VALID_RESPONSE,
        ) as mock_gemini,
    ):
        result, attempts = llm.generate_response(
            "What is Article 14?",
            ["Article 14 guarantees equality."],
            "",
        )

    assert result == json.loads(VALID_RESPONSE)
    assert attempts == 1
    mock_groq.assert_called_once()
    mock_gemini.assert_called_once()


def test_retry_loop_reaches_attempt_three():
    invalid_response_1 = '{"answer": "Missing classification"}'
    invalid_response_2 = (
        '{"answer": "Test answer", "is_follow_up": "false"}'
    )
    valid_response = json.dumps(
        {
            "answer": "Test answer",
            "is_follow_up": False,
        }
    )

    with (
        patch("llm.LLM_PROVIDER", llm.GROQ),
        patch(
            "llm.generate_groq_response",
            side_effect=[
                invalid_response_1,
                invalid_response_2,
                valid_response,
            ],
        ) as mock_groq,
        patch("builtins.print") as mock_print,
    ):
        result, attempts = llm.generate_response(
            "What is Article 14?",
            ["Article 14 guarantees equality."],
            "",
        )

    assert result == {
        "answer": "Test answer",
        "is_follow_up": False,
    }
    assert attempts == 3
    assert mock_groq.call_count == 3

    third_prompt = mock_groq.call_args_list[2].args[0]
    assert "previous response failed validation" in third_prompt.lower()

    printed_output = "\n".join(
        str(call) for call in mock_print.call_args_list
    )

    assert "Attempt 1/3" in printed_output
    assert "Attempt 2/3" in printed_output
    assert "Attempt 3/3" in printed_output


def test_retry_loop_raises_after_three_invalid_responses():
    invalid_response = '{"answer": "Missing classification"}'

    with (
        patch("llm.LLM_PROVIDER", llm.GROQ),
        patch(
            "llm.generate_groq_response",
            return_value=invalid_response,
        ) as mock_groq,
    ):
        with pytest.raises(ValueError):
            llm.generate_response(
                "What is Article 14?",
                ["Article 14 guarantees equality."],
                "",
            )

    assert mock_groq.call_count == 3


def test_generate_response_rejects_invalid_provider():
    with patch("llm.LLM_PROVIDER", "unsupported"):
        with pytest.raises(
            ValueError,
            match="Unsupported LLM provider",
        ):
            llm.generate_response(
                "What is Article 14?",
                ["Article 14 guarantees equality."],
                "",
            )


def test_generate_response_propagates_invalid_json():
    with (
        patch("llm.LLM_PROVIDER", llm.OLLAMA),
        patch(
            "llm.generate_ollama_response",
            return_value="This is not JSON",
        ) as mock_ollama,
    ):
        with pytest.raises(json.JSONDecodeError):
            llm.generate_response(
                "What is Article 14?",
                ["Article 14 guarantees equality."],
                "",
            )

    assert mock_ollama.call_count == 3
