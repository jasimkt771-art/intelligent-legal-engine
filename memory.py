from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY


def connect_to_supabase():
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return client

def fetch_history(session_id):
    client = connect_to_supabase()

    response = (
        client.table("chat_history")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    return response.data

def save_turn(session_id, user_query, bot_response):
    client = connect_to_supabase()

    client.table("chat_history").insert({
        "session_id": session_id,
        "user_query": user_query,
        "bot_response": bot_response
    }).execute()

def format_history(history):
    history = list(reversed(history))

    formatted_history = ""

    for turn in history:
        formatted_history += (
            f"User: {turn['user_query']}\n"
            f"Assistant: {turn['bot_response']}\n\n"
        )

    return formatted_history

def get_memory(session_id):
    history = fetch_history(session_id)

    formatted_history = format_history(history)

    return formatted_history

if __name__ == "__main__":

    session_id = "test_session"

    # Save two conversations
    save_turn(
        session_id,
        "What is Article 21?",
        "Article 21 protects the right to life."
    )

    save_turn(
        session_id,
        "Explain Article 19.",
        "Article 19 guarantees several freedoms."
    )

    # Fetch them back
    history = fetch_history(session_id)

    print("Raw History:")
    print(history)

    # Format for the LLM
    formatted = format_history(history)

    print("\nFormatted History:")
    print(formatted)

    # Test the convenience function
    memory = get_memory(session_id)

    print("\n=== Memory from get_memory() ===")
    print(memory)