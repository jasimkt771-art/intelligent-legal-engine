import streamlit as st
import uuid
from retrieval import get_context
from memory import (get_memory, fetch_history, save_turn)
from cache import (connect_to_cache, check_cache, save_to_cache, show_exact_cache, clear_cache)
from llm import generate_response

def initialize_connections():
    redis_client, cache_index = connect_to_cache()
    return redis_client, cache_index

def load_chat_history(session_id):
    history = fetch_history(session_id)

    history.reverse()

    for turn in history:
        with st.chat_message("user"):
            st.write(turn["user_query"])

        with st.chat_message("assistant"):
            st.write(turn["bot_response"])

def process_query(query, session_id, redis_client, cache_index):
    response = check_cache(redis_client, cache_index, query)

    if response:
        print(f'\nCached\n'
              f'Query:\n{query}\n\n'
              f'WITH\n\nResponse:\n{response}\n\n')
        return response

    contexts = get_context(query)

    history = get_memory(session_id)

    response = generate_response(query, contexts, history)
    print("\nResponse is from LLLM\n")

    save_turn(session_id, query, response)

    print(f'Saved\nQuery:\n{query}\n\nAND\n\nResponse:\n{response}\n\nTo database chat_history with session_id: {session_id}\n\n')
    save_to_cache(redis_client, cache_index, query, response)
    print()
    return response

def main():
    st.set_page_config(
        page_title="Intelligent Legal Engine",
        page_icon="⚖️",
        layout="wide"
    )

    st.title("⚖️ Intelligent Legal Engine")

    redis_client, cache_index = initialize_connections()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        print('\n=======New Conversation======')
        print("New Session id: ", st.session_state.session_id)

    else:
        session_id = st.session_state.session_id
        print('\n=======New Sub-Conversation======')
        print("Current Session id: ", session_id)

    with st.sidebar:
        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button("+ New Chat"):
                st.session_state.session_id = str(uuid.uuid4())
                print('\n=======New Conversation(Button)======')
                print("New Session id: ", st.session_state.session_id)

        with col2:
            st.button("⚙️")

        st.divider()

    load_chat_history(session_id)

    query = st.chat_input("Ask a legal question...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        response = process_query(
            query,
            session_id,
            redis_client,
            cache_index
        )

        with st.chat_message("assistant"):
            st.write(response)


if __name__ == "__main__":
    main()