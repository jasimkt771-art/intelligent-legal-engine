import ollama
from google import genai
from config import LLM_PROVIDER, OLLAMA_MODEL, GEMINI_MODEL, GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

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
- Use bullet points or numbered lists when they improve clarity.
- Avoid long blocks of text.
- Keep related ideas together.

Answer:
"""

        return prompt

    except TypeError as e:
        print('\nInvalid input for prompt construction(build_prompt[llm.py]): \n', e)
        raise

    except Exception as e:
        print(f"\nError building prompt(build_prompt[llm.py]): \n{e}")
        raise

def print_prompt_preview(query, contexts, history, max_chars=100):
    #Print a readable, truncated preview of the prompt for terminal debugging.
    print("\n========== PROMPT PREVIEW ==========\n")
    print(f"User Question:\n{query}\n")
    print(f"Previous Conversation:\n{history}\n")
    print("Retrieved Context (truncated for display):")
    for i, ctx in enumerate(contexts, start=1):
        print(f"{ctx[:max_chars]}{'...' if len(ctx) > max_chars else ''}")
        print()
    print("(Full context was sent to the LLM.)")
    print("Answer: ")

def generate_response(query, contexts, history):
    try:
        prompt = build_prompt(query, contexts, history)

        # Display-only preview — full context still goes to the LLM
        print_prompt_preview(query, contexts, history, max_chars=200)

        if LLM_PROVIDER == "ollama":

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

        elif LLM_PROVIDER == "gemini":

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

        else:
            raise ValueError(
                f"Unsupported LLM provider(generate_response[llm.py]): {LLM_PROVIDER}"
            )

        return response["message"]["content"]

    except KeyError as e:
        print(
            "\nMissing required field in Ollama response(generate_response[llm.py]): \n",
            e
        )
        raise

    except Exception as e:
        print(
            '\nError generating response(generate_response[llm.py]):\n',
            e
        )
        raise

""""
Format of response object:
{
    "model": "llama3.1",
    "created_at": "...",
    "message": {
        "role": "assistant",
        "content": "Article 21 protects life and personal liberty."
    },
    "done": True
}
"""

def main():
    query = "What fundamental rights protect me against unlawful arrest?"

    contexts = [
        "Article 21: No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "Article 22: Provides safeguards against arbitrary arrest and detention.",
        "Article 21: No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "Article 22: Provides safeguards against arbitrary arrest and detention.",
        "Article 21: No person shall be deprived of his life or personal liberty except according to procedure established by law.",
    ]

    history = """
User: What is Article 21?
Assistant: Article 21 guarantees protection of life and personal liberty.
"""

    print("========== GENERATED PROMPT ==========\n")

    prompt = build_prompt(query, contexts, history)
    # print(prompt)

    print("\n========== LLM RESPONSE ==========\n")

    response = generate_response(query, contexts, history)
    print(response)


if __name__ == "__main__":
    main()