import ollama
from config import OLLAMA_MODEL

"""cntxt1 = contexts[0][:200]
cntxt2 = contexts[1][:200]
cntxt3 = contexts[2][:200]
cntxt4 = contexts[3][:200]
cntxt5 = contexts[4][:200]"""

def build_prompt(query, contexts, history):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not isinstance(contexts, list):
            raise TypeError("Query must be a list.")

        if not isinstance(history, str):
            raise TypeError("History must be a string.")

        context_text = "\n\n".join(context[:200] for context in contexts)

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
    - Write every sentence on its own line.
    - Never place two sentences on the same line.
    - Insert a newline after every period, question mark, or exclamation mark.
    
    Answer:
    """

        return prompt

    except TypeError as e:
        print('Invalid input for prompt construction(build_prompt[llm.py]): \n', e)
        raise

    except Exception as e:
        print(f"Error building prompt(build_prompt[llm.py]): \n{e}")
        raise

def generate_response(query, contexts, history):
    try:
        prompt = build_prompt(query, contexts, history)

        print("\n========== PROMPT ==========\n")
        split = prompt.split("Instructions:", 1)[0]
        print(split)
        print('Answer: ')

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

    except KeyError as e:
        print("Missing required field in Ollama response(generate_response[llm.py]): \n",e)
        raise

    except Exception as e:
        print('Error generating response(generate_response[llm.py]): \n',e)
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