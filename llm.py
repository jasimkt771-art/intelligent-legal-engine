import ollama
from config import OLLAMA_MODEL, LLM_PROVIDER
from retrieval import get_context
from memory import get_memory
from cache import check_cache

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
- Answer only using the retrieved context.
- Cite the relevant Article whenever possible.
- If the answer cannot be found in the context, say you don't know.

Answer:
"""

    return prompt

def generate_response(query, contexts, history):
    prompt = build_prompt(query, contexts, history)

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