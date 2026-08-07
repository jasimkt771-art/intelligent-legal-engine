import streamlit as st
import uuid
from retrieval import get_context
from memory import (get_memory, fetch_history, save_turn)
from cache import (connect_to_redis, connect_to_cache_index, check_cache, save_to_cache, show_exact_cache, clear_cache)
from llm import generate_response

def initialize_connections():
    redis_client = connect_to_redis()
    cache_index = connect_to_cache_index()

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
        return response

    contexts = get_context(query)

    history = get_memory(session_id)

    response = generate_response(query, contexts, history)

    save_turn(session_id, query, response)

    save_to_cache(redis_client, cache_index, query, response)

    return response

def main():
    st.set_page_config(
        page_title="Intelligent Legal Engine",
        page_icon="⚖️",
        layout="wide"
    )

    st.title("⚖️ Intelligent Legal Engine")

    redis_client, cache_index = initialize_connections()

    with st.sidebar:
        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button("+ New Chat"):
                st.session_state.session_id = str(uuid.uuid4())

        with col2:
            st.button("⚙️")

        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        session_id = st.session_state.session_id

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