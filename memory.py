from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY


def connect_to_supabase():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return client

    except Exception as e:
        print(f"Error connecting to Supabase(connect_to_supabase[memory.py]): \n{e}")
        raise


def fetch_history(session_id):
    try:
        if not isinstance(session_id, str):
            raise TypeError("Session ID must be a string.")

        client = connect_to_supabase()

        response = (
            client.table("chat_history")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )

        if response.data is None:
            return []

        return response.data

    except TypeError as e:
        print(f"Invalid session ID for history retrieval(fetch_history[memory.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error fetching chat history(fetch_history[memory.py]): \n{e}")
        raise


def save_turn(session_id, user_query, bot_response):
    try:
        if not isinstance(session_id, str):
            raise TypeError("Session ID must be a string.")

        if not isinstance(user_query, str):
            raise TypeError("User query must be a string.")

        if not isinstance(bot_response, str):
            raise TypeError("Bot response must be a string.")

        client = connect_to_supabase()

        client.table("chat_history").insert({
            "session_id": session_id,
            "user_query": user_query,
            "bot_response": bot_response
        }).execute()

    except TypeError as e:
        print(f"Invalid input for saving chat turn(save_turn[memory.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error saving chat turn(save_turn[memory.py]): \n{e}")
        raise


def format_history(history):
    try:
        if not isinstance(history, list):
            raise TypeError("History must be a list.")

        if not history:
            return ""

        history = list(reversed(history))

        formatted_history = ""

        for turn in history:
            try:
                user_query = turn["user_query"]
                bot_response = turn["bot_response"]

            except KeyError as e:
                print(f"Missing required history field: \n{e}")
                raise

            formatted_history += (
                f"User: {user_query}\n"
                f"Assistant: {bot_response}\n\n"
            )

        return formatted_history

    except TypeError as e:
        print(f"Invalid history input(format_history[memory.py]): \n{e}")
        raise

    except Exception as e:
        print(f"Error formatting chat history(format_history[memory.py]): \n{e}")
        raise


def get_memory(session_id):
    try:
        history = fetch_history(session_id)

        formatted_history = format_history(history)

        return formatted_history

    except Exception as e:
        print(f"Error getting memory(get_memory[memory.py]): \n{e}")
        raise


def main():
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


if __name__ == "__main__":
    main()