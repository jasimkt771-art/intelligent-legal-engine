import streamlit as st
import uuid
import supabase

from retrieval import get_context
from memory import get_memory, fetch_session_messages, get_recent_sessions, save_turn
from cache import connect_to_cache, check_cache, save_to_cache, clear_cache
from llm import generate_response


# Main Streamlit application.
# Handles the user interface, conversation flow, memory, caching, retrieval, and LLM response generation.


def initialize_connections():
    try:
        redis_client, cache_index = connect_to_cache()
        return redis_client, cache_index
    except Exception as e:
        print("\nConnection error(initialize_connections[app.py]):\n", e)
        raise


def load_chat_history(session_id):
    try:
        if not isinstance(session_id, str):
            raise TypeError("Session id must be a string.")

        history = fetch_session_messages(session_id)

        if not isinstance(history, list):
            raise TypeError("History must be a list.")

        for turn in history:
            with st.chat_message("user"):
                st.write(turn["user_query"])

            with st.chat_message("assistant"):
                st.write(turn["bot_response"])

    except TypeError as e:
        print("\nInvalid input(load_chat_history[app.py]):\n", e)
        raise

    except KeyError as e:
        print("\nMissing required field in chat history(load_chat_history[app.py]):\n", e)
        raise

    except Exception as e:
        print("\nError loading chat history(load_chat_history[app.py]):\n", e)


def is_follow_up_query(query, has_history):
    """
    Determines whether a query is an obvious conversational follow-up.

    Follow-up queries should bypass the global cache because their meaning
    may depend on previous conversation history.
    """

    if not has_history:
        return False

    follow_up_patterns = [
        # References / what they said
        "what about that", "what about this", "what about it", "how about that",
        "how about this", "how about it", "what you said", "what you mentioned",
        "as you mentioned", "as you said", "you mentioned earlier", "you said earlier",

        # Meaning / significance
        "what does that mean", "what does this mean", "what does it mean",
        "what did that mean", "what did this mean", "what did it mean",
        "what is its significance", "what's its significance",
        "why is that significant", "why is this significant",
        "what does it signify", "what does that signify", "what does this signify",
        "what does that actually mean", "what exactly did you mean by that",
        "what exactly did you mean",

        # Explanation / clarification
        "explain that", "explain this", "explain it", "explain simply", "explain again",
        "explain the above", "explain the previous", "explain what you mean",
        "explain that again", "can you explain that", "can you explain this",
        "can you explain it", "can you clarify", "clarify that", "clarify this",
        "clarify it", "can you elaborate", "elaborate on that", "elaborate on this",
        "can you expand on that", "can you explain what you just said",

        # Previous answer / previous point
        "the previous answer", "the previous point", "the previous one", "the previous",
        "the above", "the above answer", "the above point", "the above one",
        "what you said earlier", "what you mentioned earlier",

        # More details
        "tell me more about that", "tell me more about this", "tell me more about it",
        "more about that", "more about this", "more about it",
        "can you tell me more about that", "can you tell me more about this",
        "give me more details about that", "give me more details about this",
        "explain further", "go into more detail about that", "go into more detail about this",
        "anything else about that", "anything else about this",

        # Application / consequences
        "what happens after that", "what happens next", "what does that do",
        "what does this do", "how does that work", "how does this work",
        "how does it work", "how does that apply", "how does this apply",
        "does that apply", "does this apply", "does it apply",
        "why does that matter", "why does this matter", "why does it matter",
        "what exactly were you referring to",

        # Comparisons / differences
        "compare that", "compare this", "compare it", "compare the above",
        "compare the previous", "how is that different", "how is this different",

        # Reasoning / why
        "why did you say", "why did you mention", "why is that", "why is this",
        "why is it", "why does that", "why does this", "why would that",
        "why would this", "how did you get that", "how did you reach that",

        # References to context
        "based on that", "based on this", "based on what you said",
        "according to that", "according to this", "from that", "from this",
        "then what", "and what about", "so what does that mean",
        "so what does this mean",

        # Informal / conversational
        "what do u mean", "what do you mean", "what did you mean",
        "what are you saying", "what are you referring to"
    ]

    normalized_query = query.lower().strip()

    return any(pattern in normalized_query for pattern in follow_up_patterns)


def build_retrieval_query(query, history):
    """
    Builds a retrieval query using the latest conversation
    and the current user query.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    if not isinstance(history, str):
        raise TypeError("History must be a string.")

    if not history.strip():
        return query

    lines = history.splitlines()
    message_starts = []

    for i in range(len(lines)):
        if lines[i].startswith(("User:", "Assistant:")):
            message_starts.append(i)

    messages = []

    for i in range(len(message_starts)):
        start = message_starts[i]
        end = message_starts[i + 1] if i + 1 < len(message_starts) else len(lines)
        messages.append("\n".join(lines[start:end]).strip())

    latest_user_message = ""
    latest_assistant_message = ""

    for message in reversed(messages):
        if message.startswith("Assistant:") and not latest_assistant_message:
            latest_assistant_message = message[len("Assistant:"):].strip()
        elif message.startswith("User:") and not latest_user_message:
            latest_user_message = message[len("User:"):].strip()

        if latest_user_message and latest_assistant_message:
            break

    retrieval_query = f"""
Previous User Question:
{latest_user_message}

Previous Assistant Answer:
{latest_assistant_message}

Current User Question:
{query}
""".strip()

    print("\n====Current Retrieval Query=====\n\n", retrieval_query)

    return retrieval_query


def process_query(query, session_id, redis_client, cache_index):
    try:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not isinstance(session_id, str):
            raise TypeError("Session id must be a string.")

        history = get_memory(session_id)
        has_history = bool(history.strip())
        is_follow_up = is_follow_up_query(query, has_history)

        # Follow-up queries bypass the global cache because their meaning depends on conversation history.
        if not is_follow_up:
            print("\nQuery is not a follow-up, checking cache\n")
            response = check_cache(redis_client, cache_index, query)

            if response:
                print(f"\nCached\nQuery:\n{query}\n\nWITH\n\nResponse:\n{response}\n\n")
                return response

        if is_follow_up:
            # Add the previous conversation to the retrieval query so references like "that" can be resolved.
            retrieval_query = build_retrieval_query(query, history)
        else:
            retrieval_query = query

        contexts = get_context(retrieval_query)

        print("\nResponse is from LLLM\n")

        response = generate_response(query, contexts, history)

        try:
            save_turn(session_id, query, response)
            print(f"Saved\nQuery:\n{query}\n\nAND\n\nResponse:\n{response}\n\nTo database chat_history with session_id: {session_id}\n")

            try:
                if not is_follow_up:
                    print("\nQuery is not a follow-up\nSaving to cache\n")
                    save_to_cache(redis_client, cache_index, query, response)
                else:
                    # Avoid caching follow-ups because their meaning depends on conversation history.
                    print("\nQuery is a follow-up, so its not being saved to cache\n")

            except Exception as e:
                print(f"Cache unavailable, response will not be saved to cache(process_query[app.py]):\n{e}")

        except supabase.SupabaseException as e:
            print(f"Supabase unavailable, response will not be saved to memory or cache(process_query[app.py]):\n{e}")

        print()
        return response

    except Exception as e:
        print(f"Error generating response(process_query[app.py]):\n{e}")
        raise


def render_sidebar():
    selected_session = None
    new_chat_clicked = False

    with st.sidebar:
        st.header("Chats")

        if st.button("+ New Chat"):
            new_chat_clicked = True

        st.divider()

        st.markdown("<h2 style='margin-top: -25px; margin-bottom: 20px; font-size: 28px;'>Previous Chats</h2>", unsafe_allow_html=True)

        try:
            sessions = get_recent_sessions(limit=5)

            for session in sessions:
                session_id = session["session_id"]
                first_query = session["first_query"]

                if st.button(first_query, key=f"chat_{session_id}", use_container_width=True):
                    selected_session = session_id

        except Exception as e:
            print(f"\nError loading recent chats(render_sidebar[app.py]):\n{e}")
            st.error("Could not load previous chats.")

    return new_chat_clicked, selected_session


def main():
    try:
        st.set_page_config(page_title="Intelligent Legal Engine", page_icon="⚖️", layout="wide")
        st.title("⚖️ Intelligent Legal Engine")

        redis_client, cache_index = initialize_connections()

        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            print("\n=======New Conversation======")
            print("New Session id:", st.session_state.session_id)

        new_chat_clicked, selected_session = render_sidebar()

        if new_chat_clicked:
            st.session_state.session_id = str(uuid.uuid4())
            print("\n=======New Conversation(Button)======")
            print("New Session id:", st.session_state.session_id)
            st.rerun()

        elif selected_session:
            st.session_state.session_id = selected_session
            print("\n\n=======Session Selected======")
            print("Selected Session id:", st.session_state.session_id)
            st.rerun()

        session_id = st.session_state.session_id

        print("\n=======New Sub-Conversation======")
        print("Current Session id:", session_id)

        load_chat_history(session_id)

        query = st.chat_input("Ask a legal question...")

        if query:
            with st.chat_message("user"):
                st.write(query)

            try:
                response = process_query(query, session_id, redis_client, cache_index)

            except Exception as e:
                print(f"Query processing failed(main[app.py]):\n{e}")
                st.error("Something went wrong while processing your question. Please try again.")
                return

            with st.chat_message("assistant"):
                st.write(response)

    except Exception as e:
        print(f"Application error(main[app.py]):\n{e}")
        st.error("The application encountered an error. Please try again later.")
        return


if __name__ == "__main__":
    main()