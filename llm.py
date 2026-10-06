import ollama
from google import genai
from groq import Groq

from config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    GEMINI_MODEL,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI,
    GROQ,
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
- Respond exactly: "The answer cannot be determined from the retrieved context."

If the user asks about an Article that is not present in the Retrieved Context:
- Say that the requested Article was not included in the retrieved context.

Answer:
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


def generate_gemini_response(prompt):
    try:
        print("Model == GEMINI")

        gemini_prompt = (
            prompt
            + """
Gemini-Specific Instructions:
- Explain the answer as if you are speaking to someone with no legal background.
- Use simple, everyday language instead of formal legal language.
- After mentioning a legal provision or legal term, explain what it means in simple words.
- Focus on explaining what the provision actually means in practical terms in relation to the user's question.
- Avoid unnecessary legal jargon.
- If a legal term is necessary, immediately explain it in simple words.
- Use clear and practical examples whenever they help the user understand the provision.
- Keep the response concise while still covering all relevant information from the Retrieved Context.
"""
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=gemini_prompt
        )

        return response.text

    except Exception as e:
        print(f"\nGemini failed:\n{e}")
        raise


def generate_groq_response(prompt):
    try:
        print("\nModel == GROQ")

        groq_prompt = (
            prompt
            + """
Groq-Specific Instructions:
- Explain the answer as if you are speaking to someone with no legal background.
- Use simple, everyday language instead of formal legal language.
- After mentioning a legal provision or legal term, explain what it means in simple words.
- Focus on explaining what the provision actually means in practical terms in relation to the user's question.
- Avoid unnecessary legal jargon.
- If a legal term is necessary, immediately explain it in simple words.
- Use clear and practical examples whenever they help the user understand the provision.
- Keep the response concise while still covering all relevant information from the Retrieved Context.
"""
        )

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=[{"role": "user", "content": groq_prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"\nGroq failed:\n{e}")
        raise


def generate_response(query, contexts, history):
    try:
        prompt = build_prompt(query, contexts, history)

        print_prompt_preview(query, contexts, history, max_chars=200)

        if LLM_PROVIDER == "ollama":
            print("Model == OLLAMA")

            ollama_prompt = (
                prompt
                + """
Ollama-Specific Instructions:
- Provide a longer and more detailed response while remaining accurate and grounded in the Retrieved Context.
"""
            )

            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": ollama_prompt}],
            )

            return response["message"]["content"]

        elif LLM_PROVIDER == GROQ:
            try:
                return generate_groq_response(prompt)

            except Exception as e:
                # Use Gemini as a fallback if the primary Groq request fails.
                print(f"\nGroq unavailable. Switching to Gemini.\nGroq error: {e}\n")

                return generate_gemini_response(prompt)

        elif LLM_PROVIDER == GEMINI:
            return generate_gemini_response(prompt)

        else:
            raise ValueError(
                f"Unsupported LLM provider(generate_response[llm.py]): {LLM_PROVIDER}"
            )

    except KeyError as e:
        print(
            "\nMissing required field in LLM response(generate_response[llm.py]): \n", e
        )
        raise

    except Exception as e:
        print("\nError generating response(generate_response[llm.py]):\n", e)
        raise
