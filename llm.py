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
    GROQ
)


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
- U can mention the relevant article's/article clause's content and then explain what it actually says in simple words in relevance to the query. Use examples whenever possible.

Grounding Rules:
- The Retrieved Context is the only source of truth for answering the user's question.
- Use only information explicitly present in the Retrieved Context.
- Do not use your own knowledge, training knowledge, assumptions, or outside sources to answer the question.
- For each Article in the Retrieved Context, determine whether it is relevant to the user's question.
- Use only the relevant Articles to answer the question.
- Do not mention, cite, or introduce any Article that does not appear in the Retrieved Context.
- Do not invent or introduce facts, legal provisions, Articles, cases, rules, acts, or other information that is not explicitly present in the Retrieved Context.

If the Retrieved Context contains enough information to answer the question:
- Answer using only the relevant retrieved information.
- Explain why the relevant Article/Articles answer the question.
- Explain in detail even when u have low data like the response must be big but not inaccurate.
- Provide enough explanation to fully answer the question without adding unsupported information.
- The response length should follow the Response Length rules above.
- Cite the relevant Article naturally.
- At the end, u should have a section named citations that shows all the articles in the Retrieved Context that are used to answer the query.
- It should look like this:
  Citations:
  - Highest relevant article
  - 2nd Highest relevant article
  and so on. Rank according to relevance to the query. Only mention the articles in the Retrieved Context
- Use this format only for queries that requires multiple articles to be answered.
- For queries that only talk about a single article eg: What is Article 21? or What rights do Article 21 provide me? only mention the relevant article in this case Article 21 even when the retrieved context contains multiple articles.

If the Retrieved Context does not contain enough information to answer the question:
- Respond exactly: "The answer cannot be determined from the retrieved context."
- Do not provide an answer using outside knowledge.
- Do not mention or cite Articles that are not present in the Retrieved Context.
- Do not rank the Articles ie dont show the articles in citations section.

If the user asks about an Article that is not present in the Retrieved Context:
- Say that the requested Article was not included in the retrieved context.

Formatting Rules:
- Make the response easy to read.
- Use short paragraphs.
- Use bullet points or numbered lists when they improve readability.
- Avoid long blocks of text.
- Keep related ideas together.

Answer:
"""

        return prompt

    except TypeError as e:
        print(
            '\nInvalid input for prompt construction(build_prompt[llm.py]): \n',
            e
        )
        raise

    except Exception as e:
        print(
            f"\nError building prompt(build_prompt[llm.py]): \n{e}"
        )
        raise


def print_prompt_preview(query, contexts, history, max_chars=100):
    print("\n========== PROMPT PREVIEW ==========\n")
    print(f"User Question:\n{query}\n")
    print(f"Previous Conversation:\n{history}\n")
    print("Retrieved Context (truncated for display):")

    for i, ctx in enumerate(contexts, start=1):
        print(
            f"{ctx[:max_chars]}"
            f"{'...' if len(ctx) > max_chars else ''}"
        )
        print()

    print("(Full context was sent to the LLM.)")
    print("Answer: ")


def generate_gemini_response(prompt):
    try:
        print("Model == GEMINI")

        gemini_prompt = prompt + """
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

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=gemini_prompt
        )

        return response.text

    except Exception as e:
        print(
            f"\nGemini failed:\n{e}"
        )
        raise


def generate_groq_response(prompt):
    try:
        print("Model == GROQ")

        groq_prompt = prompt + """
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

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": groq_prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(
            f"\nGroq fallback failed: {e}"
        )
        raise


def generate_response(query, contexts, history):
    try:
        prompt = build_prompt(query, contexts, history)

        print_prompt_preview(
            query,
            contexts,
            history,
            max_chars=200
        )

        if LLM_PROVIDER == "ollama":
            print("Model == OLLAMA")

            ollama_prompt = prompt + """
Ollama-Specific Instructions:
- Provide a longer and more detailed response while remaining accurate and grounded in the Retrieved Context.
"""

            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": ollama_prompt
                    }
                ]
            )

            return response["message"]["content"]

        elif LLM_PROVIDER == GEMINI:

            try:
                return generate_gemini_response(prompt)

            except Exception as e:
                print(
                    f"\nGemini unavailable. Switching to Groq.\n"
                    f"Gemini error: {e}\n"
                )

                return generate_groq_response(prompt)

        elif LLM_PROVIDER == GROQ:
            return generate_groq_response(prompt)

        else:
            raise ValueError(
                f"Unsupported LLM provider(generate_response[llm.py]): "
                f"{LLM_PROVIDER}"
            )

    except KeyError as e:
        print(
            "\nMissing required field in LLM response"
            "(generate_response[llm.py]): \n",
            e
        )
        raise

    except Exception as e:
        print(
            '\nError generating response(generate_response[llm.py]):\n',
            e
        )
        raise