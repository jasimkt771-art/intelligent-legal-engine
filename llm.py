import ollama
from config import OLLAMA_MODEL

def build_prompt(query, contexts, history):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a legal assistant.

Previous Conversation:
{history}

Retrieved Context:
{context_text}

User Question:
{query}

Instructions:
- Answer naturally and conversationally.
- Imagine you're explaining the answer to someone in a ChatGPT conversation.
- Do not sound like a search engine or legal document.
- Start with the direct answer.
- Keep the answer simple and easy to understand.

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
- Cite the relevant Article naturally.

If the Retrieved Context does not contain enough information to answer the question:
- Respond exactly: "The answer cannot be determined from the retrieved context."
- Do not provide an answer using outside knowledge.
- Do not mention or cite Articles that are not present in the Retrieved Context.
- Do not rank the Articles.

If the user asks about an Article that is not present in the Retrieved Context:
- Say that the requested Article was not included in the retrieved context.

Formatting Rules:
- Write every sentence on its own line.
- Never place two sentences on the same line.
- Insert a newline after every period, question mark, or exclamation mark.

Answer:
"""

    return prompt

def generate_response(query, contexts, history):
    prompt = build_prompt(query, contexts, history)

#    print("\n========== PROMPT ==========\n")
#    print(prompt)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

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

if __name__ == "__main__":
    query = "What fundamental rights protect me against unlawful arrest?"

    contexts = [
        "Article 21: No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "Article 22: Provides safeguards against arbitrary arrest and detention."
    ]

    history = """
User: What is Article 21?
Assistant: Article 21 guarantees protection of life and personal liberty.
"""

    print("========== GENERATED PROMPT ==========\n")

    prompt = build_prompt(query, contexts, history)
    print(prompt)

    print("\n========== LLM RESPONSE ==========\n")

    response = generate_response(query, contexts, history)
    print(response)