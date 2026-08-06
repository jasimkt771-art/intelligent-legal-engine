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

if __name__ == "__main__":

    redis_client, cache_index = initialize_connections()

    session_id = "test_session"

    while True:

        print("\n========== Intelligent Legal Engine ==========")
        print("1. Ask Question")
        print("2. Show Exact Cache")
        print("3. Clear Cache")
        print("4. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":

            query = input("\nEnter your question: ")

            response = process_query(
                query,
                session_id,
                redis_client,
                cache_index
            )

            print("\nAssistant:\n")
            print(response)

        elif choice == "2":

            show_exact_cache(redis_client)

        elif choice == "3":

            clear_cache(
                redis_client,
                cache_index
            )

            print("\nCache cleared successfully.")

        elif choice == "4":

            print("Goodbye!")
            break

        else:

            print("\nInvalid choice.")