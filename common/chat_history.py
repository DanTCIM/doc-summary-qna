import streamlit as st
import pandas as pd


def display_chat_history(msgs):
    tmp_query = ""
    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        if msg.content.startswith("Query:"):
            tmp_query = msg.content.lstrip("Query: ")
        elif msg.content.startswith("# Retrieval"):
            with st.expander(f"📖 **Context Retrieval:** {tmp_query}", expanded=False):
                st.write(msg.content, unsafe_allow_html=True)
        else:
            tmp_query = ""
            st.chat_message(avatars[msg.type]).write(msg.content)


def clear_chat_history(msgs):
    msgs.clear()
    msgs.add_ai_message("Welcome to actuarial document summarizer and Q&A tool!")


def convert_df(msgs):
    df = []
    for msg in msgs.messages:
        df.append({"type": msg.type, "content": msg.content})

    df = pd.DataFrame(df)
    return df.to_csv().encode("utf-8")
