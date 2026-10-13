import ollama
from google import genai
from groq import Groq
import json

from config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    GEMINI_MODEL,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI,
    GROQ,
    OLLAMA
)


# Handles LLM response generation.
# Supports Ollama for local development, Groq as the primary production provider, and Gemini as a fallback.


# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


def build_prompt(query, contexts, history):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if not isinstance(contexts, list):
            raise TypeError("Query must be a list.")
        if not isinstance(history, str):
            raise TypeError("History must be a string.")

        context_text = "\n\n".join(contexts)

        prompt = f"""
You are a legal assistant.

User Question:
{query}

Previous Conversation:
{history}

Retrieved Context:
{context_text}

Instructions:
- Answer naturally and conversationally.
- Imagine you're explaining the answer to someone in a ChatGPT conversation.
- Do not sound like a search engine or legal document.
- Start with the direct answer.
- Keep the answer simple and easy to understand.
- You can mention the relevant Article's/article clause's content and explain what it means in simple words, using examples when possible.

Grounding Rules:
- The Retrieved Context is the only source of truth for legal information.
- Use only information explicitly present in the Retrieved Context.
- Do not use your own knowledge, assumptions, or outside sources for legal claims.
- For each Article in the Retrieved Context, determine whether it is relevant to the user's question.
- Do not mention, cite, or introduce Articles that are irrelevant or not present in the Retrieved Context.
- Do not invent legal facts, provisions, Articles, cases, rules, acts, or other unsupported information.

Handling Follow-up Questions:
- First determine whether the user's message is a follow-up to the previous conversation.
- If it is a follow-up, use the previous conversation and Retrieved Context together to understand what the user is referring to.
- If the Retrieved Context contains enough information to answer the follow-up, answer it using only that information.
- A follow-up does NOT need to directly mention an Article. For example, questions such as "What does that mean?", "Why is that important?", or "Can you explain that?" should be answered using the relevant preceding context when possible.
- If the message is a follow-up but the Retrieved Context does not contain enough information to answer the specific follow-up, give a helpful conversational response based only on what is available. Do not invent missing legal information.
- If the follow-up is clearly asking for something outside the available legal context, briefly explain that the requested information is outside the available context.

Handling Normal / Non-Follow-up Questions:
- If the question can be answered from the Retrieved Context, answer it using the relevant information.
- If the question is outside the Retrieved Context or outside the legal scope of the system, clearly say that it is outside the available context/scope.
- For casual messages such as "hi", greetings, or ordinary conversation, respond naturally and conversationally rather than saying that the answer cannot be determined from the context.
- Do not invent legal information when responding to general conversation.

If the Retrieved Context contains enough information to answer a legal question:
- Explain why the relevant Article/Articles answer the question.
- Provide enough explanation to fully answer the question without adding unsupported information.
- Cite the relevant Article naturally.
- For all queries, at the end of the response, add:

Citations:
- Highest relevant article
- 2nd highest relevant article
- and so on, ranked by relevance.


If the Retrieved Context does not contain enough information for a non-follow-up legal question:
- Set the "answer" field to exactly:
  "The answer cannot be determined from the retrieved context."

If the user asks about an Article that is not present in the Retrieved Context:
- Say that the requested Article was not included in the retrieved context.

Follow-up classification:
- Set "is_follow_up" to true if the current query depends on,
  refers to, clarifies, or continues the previous conversation.
- Set "is_follow_up" to false if the current query is independent.
- A related legal topic alone does not make a query a follow-up.
- If Previous Conversation is empty, set "is_follow_up" to false.

Output requirements:
- Return one JSON object containing exactly these two fields:
  "answer" and "is_follow_up".
- "answer" must contain the complete user-facing answer.
- "is_follow_up" must be a Boolean: true or false.
- Do not put the Boolean inside the answer text.
- Do not include Markdown fences or text outside the JSON object.

Required structure:
{{"answer": "Your complete answer here", "is_follow_up": true}}
"""


        return prompt

    except TypeError as e:
        print("\nInvalid input for prompt construction(build_prompt[llm.py]): \n", e)
        raise

    except Exception as e:
        print(f"\nError building prompt(build_prompt[llm.py]): \n{e}")
        raise


def print_prompt_preview(query, contexts, history, max_chars=100):
    print("\n========== PROMPT PREVIEW ==========\n")
    print(f"User Question:\n{query}\n")
    print(f"Previous Conversation:\n{history}\n")
    print("Retrieved Context (truncated for display):")

    for i, ctx in enumerate(contexts, start=1):
        print(f"{ctx[:max_chars]}{'...' if len(ctx) > max_chars else ''}")
        print()

    print("(Full context was sent to the LLM.)")
    print("Answer: ")




def generate_groq_response(prompt):
    try:
        print("\nModel == GROQ\n")

        groq_prompt = (
            prompt
            + """
Groq-Specific Instructions:
- Explain the answer in simple, everyday language.
- Explain legal terms in plain language.
- Use practical examples when helpful.
- Keep the answer concise but complete.
- Follow the required JSON output structure.
"""
        )

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": groq_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(
            "\nError generating Groq response"
            "(generate_groq_response[llm.py]):\n",
            e,
        )
        raise


'''
Object hierarchy — Groq response fields accessed by this function:

response
└── choices
    └── list
        └── choices[0]
            └── message
                └── content
                    └── string containing JSON
                        ├── "answer"
                        └── "is_follow_up"

Exact access:
response.choices[0].message.content

Explanation:
- response is the API response object.
- choices is the list of generated choices.
- choices[0] selects the first generated choice.
- message contains the model's message.
- content contains the raw JSON text.
- The JSON fields are not accessed here. They are parsed later
  by parse_llm_response().
'''

def generate_gemini_response(prompt):
    try:
        print("\nModel == GEMINI")

        gemini_prompt = (
            prompt
            + """
Gemini-Specific Instructions:
- Explain the answer in simple, everyday language.
- Explain legal terms in plain language.
- Use practical examples when helpful.
- Keep the answer concise but complete.
- Follow the required JSON output structure.
"""
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=gemini_prompt,
            config={
                "response_mime_type": "application/json"
            },
        )

        if response.text is None:
            raise ValueError(
                "Gemini returned no response text."
            )

        return response.text

    except ValueError as e:
        print(
            "\nInvalid Gemini response"
            "(generate_gemini_response[llm.py]):\n",
            e,
        )
        raise

    except Exception as e:
        print(
            "\nError generating Gemini response"
            "(generate_gemini_response[llm.py]):\n",
            e,
        )
        raise


'''
Object hierarchy — Gemini response fields accessed by this function:

response
└── text
    └── string containing JSON
        ├── "answer"
        └── "is_follow_up"

Exact accesses:
response.text

Explanation:
- response is the Gemini API response object.
- text contains the generated response as a string.
- The JSON fields are not accessed at this stage.
- parse_llm_response() parses and validates those fields later.
'''


def generate_ollama_response(prompt):
    try:
        print("\nModel == OLLAMA\n")

        ollama_prompt = (
            prompt
            + "\n\nReturn only a valid JSON object with exactly "
            'two keys: "answer" and "is_follow_up". '
            '"answer" must be a non-empty string. '
            '"is_follow_up" must be a Boolean (true or false). '
            "Do not include Markdown fences or additional text."
        )

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": ollama_prompt,
                }
            ],
            format="json",
        )

        raw_response = response["message"]["content"]

        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError(
                "Ollama returned an empty or invalid response."
            )

        return raw_response

    except Exception as e:
        print(
            "\nException content/text("
            "generate_ollama_response[llm.py]):\n",
            e,
        )
        raise

'''
Relevant response hierarchy (simplified):

response
└── "message": dict
    └── "content": str
        └── JSON text
            ├── "answer": str
            └── "is_follow_up": bool

Exact access:
response["message"]["content"]

Function contract: A — raises on failure.
A successful call returns the raw JSON string.
An empty response is treated as a failure, not a legitimate result.
'''


def parse_llm_response(raw_response):
    try:
        if not isinstance(raw_response, str):
            raise TypeError("LLM response must be a string.")

        result = json.loads(raw_response)

        if not isinstance(result, dict):
            raise ValueError("LLM response must be a JSON object.")

        expected_keys = {"answer", "is_follow_up"}

        if set(result.keys()) != expected_keys:
            raise ValueError(
                "LLM response must contain exactly "
                "'answer' and 'is_follow_up'."
            )

        answer = result["answer"]
        is_follow_up = result["is_follow_up"]

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")

        if not isinstance(is_follow_up, bool):
            raise ValueError("'is_follow_up' must be a Boolean.")

        return {
            "answer": answer,
            "is_follow_up": is_follow_up,
        }

    except json.JSONDecodeError as e:
        print("\nInvalid JSON(parse_llm_response[llm.py]):\n", e)
        raise

    except TypeError as e:
        print("\nInvalid input(parse_llm_response[llm.py]):\n", e)
        raise

    except ValueError as e:
        print("\nInvalid response structure(parse_llm_response[llm.py]):\n", e)
        raise

    except Exception as e:
        print("\nError parsing LLM response(parse_llm_response[llm.py]):\n", e)
        raise




def generate_response(query, contexts, history):
    """
    Generate and validate an LLM response.

    Contract:
    A — returns a validated dictionary or raises an exception.

    A successful result contains:
    {
        "answer": str,
        "is_follow_up": bool
    }

    Invalid response formats are retried up to three total attempts.
    """

    max_attempts = 3

    try:
        prompt = build_prompt(query, contexts, history)

        current_prompt = prompt

        print("\nPrompt preview:\n", prompt[:1000])

        for attempt in range(1, max_attempts + 1):
            print(
                f"\nAttempt {attempt}/{max_attempts}: "
                "Generating LLM response..."
            )

            raw_response = None

            # Generate using the configured provider.
            if LLM_PROVIDER == OLLAMA:
                raw_response = generate_ollama_response(
                    current_prompt
                )

            elif LLM_PROVIDER == GROQ:
                try:
                    raw_response = generate_groq_response(
                        current_prompt
                    )

                except Exception as e:
                    print(
                        "\nGroq unavailable. Switching to Gemini"
                        "(generate_response[llm.py]):\n",
                        e,
                    )

                    raw_response = generate_gemini_response(
                        current_prompt
                    )

            elif LLM_PROVIDER == GEMINI:
                raw_response = generate_gemini_response(
                    current_prompt
                )

            else:
                raise ValueError(
                    "Unsupported LLM provider"
                    "(generate_response[llm.py]): "
                    f"{LLM_PROVIDER}"
                )

            # Validate the JSON and its required fields.
            try:
                result = parse_llm_response(raw_response)

                print(
                    f"Attempt {attempt}/{max_attempts}: "
                    "Response validated successfully.\n"
                )

                return result, attempt

            except (ValueError, TypeError) as e:
                print(
                    f"\nAttempt {attempt}/{max_attempts}: "
                    "Invalid response format"
                    "(generate_response[llm.py]):\n",
                    e,
                )

                if attempt == max_attempts:
                    print(
                        "\nMaximum attempts reached. "
                        "Unable to obtain a valid response"
                        "(generate_response[llm.py])."
                    )
                    raise

                # Give the model specific instructions for the next attempt.
                current_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous response failed "
                    "validation.\n"
                    + f"Validation error: {e}\n"
                    + "Regenerate the complete answer. Return only "
                    "one valid JSON object containing exactly these "
                    "two keys:\n"
                    + '- "answer": a non-empty string\n'
                    + '- "is_follow_up": a JSON Boolean, true or false\n'
                    + "Do not put the Boolean in quotation marks. "
                    "Do not use Markdown fences or additional text."
                )

                print(
                    f"Retrying with corrective instructions: "
                    f"attempt {attempt + 1}/{max_attempts}."
                )

    except (TypeError, ValueError, json.JSONDecodeError) as e:
        print(
            "\nException content/text(generate_response[llm.py]):\n",
            e,
        )
        raise

    except Exception as e:
        print(
            "\nException content/text(generate_response[llm.py]):\n",
            e,
        )
        raise
