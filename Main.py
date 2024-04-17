## sqlite3 related (for Streamlit)
import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import streamlit as st

from common.utils import (
    StreamHandler,
    PrintRetrievalHandler,
    embeddings_model,
)

from common.prompt import summary_prompt

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.document_loaders import Docx2txtLoader
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

# from langchain_experimental.text_splitter import SemanticChunker

# API Key Setup
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

# Start Streamlit session
st.set_page_config(page_title="Doc Summary Q&A Tool", page_icon="📖")
st.header("Actuarial Document Summarizer and Q&A Tool")
st.write("Click the button in the sidebar to summarize.")
# Setup uploader
uploaded_file = st.file_uploader(
    label="Upload your own PDF, DOCX, or TXT file. Do NOT upload files that contain confidential or private information.",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=False,
    help="Pictures or charts in the document are not recognized",
)

# Initialize a session state variable to store the previous file
if "prev_file" not in st.session_state:
    st.session_state.prev_file = None

# Initialize a session state variable to store the vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Initialize a session state variable to store the text
if "loaded_doc" not in st.session_state:
    st.session_state.loaded_doc = None


if uploaded_file is not None:
    if uploaded_file != st.session_state.prev_file:
        with st.spinner("Extracting text and converting to embeddings..."):
            if uploaded_file.name.endswith(".pdf"):
                # Save the uploaded file to a temporary location
                with open("tempXXX.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = PDFMinerLoader("tempXXX.pdf", concatenate_pages=True)

            elif uploaded_file.name.endswith(".docx"):
                # Save the uploaded DOCX file to a temporary location
                with open("tempXXX.docx", "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = Docx2txtLoader("tempXXX.docx")

            elif uploaded_file.name.endswith(".txt"):
                # Save the uploaded DOCX file to a temporary location
                with open("tempXXX.txt", "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = TextLoader("tempXXX.txt")

            else:
                st.error(
                    "Unsupported file format. Please upload a file in a supported format."
                )

            st.session_state.loaded_doc = loader.load()

            # text_splitter = SemanticChunker(embeddings_model)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )

            splits = text_splitter.split_documents(st.session_state.loaded_doc)

            # Create a Chroma vector database from the document splits
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings_model,
            )
        st.session_state.prev_file = uploaded_file

# LLM flag for augmented generation (the flag only applied to llm, not embedding model)
USE_Anthropic = True

if USE_Anthropic:
    model_name = "claude-3-sonnet-20240229"
else:
    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4-0125-preview"  # gpt-4 seems to be slow

## Sidebar
with st.sidebar:
    st.header("**Actuary Helper**")
    st.write(
        f"The *{model_name}*-powered AI tool is designed to enhance the efficiency of actuaries by summarizing actuarial documents and providing answers to document-related questions. The tool uses **retrieval augmented generation (RAG)** to help Q&A."
    )
    st.write(
        "**AI's responses should not be relied upon as accurate or error-free.** The quality of the retrieved contexts and responses may depend on LLM algorithms, RAG parameters, and how questions are asked. Harness its power but **with accountability and responsibility**."
    )
    st.write(
        "Actuaries are strongly advised to **evaluate for accuracy** when using the tool. Read the retrieved contexts to compare to AI's responses. The process is built for educational purposes only."
    )

    with st.expander("⚙️ RAG Parameters"):
        num_source = st.slider(
            "Top N sources to view:", min_value=4, max_value=20, value=5, step=1
        )
        flag_mmr = st.toggle(
            "Diversity search",
            value=True,
            help="Diversity search, i.e., Maximal Marginal Relevance (MMR) tries to reduce redundancy of fetched documents and increase diversity. 0 being the most diverse, 1 being the least diverse. 0.5 is a balanced state.",
        )
        _lambda_mult = st.slider(
            "Diversity parameter (lambda):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.25,
        )
        flag_similarity_out = st.toggle(
            "Output similarity score",
            value=False,
            help="The retrieval process may become slower due to the cosine similarity calculations. A similarity score of 100% indicates the highest level of similarity between the query and the retrieved chunk.",
        )

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=msgs,
    return_messages=True,
)
# Initialize the chat history
if len(msgs.messages) == 0:
    msgs.add_ai_message("Welcome to actuarial document summarizer and Q&A tool!")

# Show the chat history
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


if uploaded_file is not None:

    # Retrieve and RAG chain
    # Create a retriever using the vector database as the search source
    search_kwargs = {"k": num_source}

    if flag_mmr:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={**search_kwargs, "lambda_mult": _lambda_mult},
        )
        # Use MMR (Maximum Marginal Relevance) to find a set of documents
        # that are both similar to the input query and diverse among themselves
        # Increase the number of documents to get, and increase diversity
        # (lambda mult 0.5 being default, 0 being the most diverse, 1 being the least)
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )  # use similarity search

    # Setup LLM and QA chain
    if USE_Anthropic:
        llm = ChatAnthropic(
            model_name=model_name,
            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"],
            temperature=0,
            streaming=True,
        )
    else:
        llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            temperature=0,
            streaming=True,
        )

    # Define LLM chain to use for summarizer
    summary_llm_chain = LLMChain(
        llm=llm, prompt=PromptTemplate.from_template(summary_prompt)
    )
    # Define StuffDocumentsChain for summary creation
    stuff_chain = StuffDocumentsChain(
        llm_chain=summary_llm_chain, document_variable_name="text"
    )

    # Define Q&A chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    def summarizer():
        with st.spinner("Summarizing the document..."):
            summary_data = stuff_chain.invoke(st.session_state.loaded_doc)[
                "output_text"
            ]
            msgs.add_ai_message(summary_data)

    st.sidebar.button(
        label="Summarize the doc (takes a minute)",
        use_container_width=True,
        on_click=summarizer,
        help="Summarizing the full document. Can take a while to complete.",
    )

    # Ask the user for a question
    if user_query := st.chat_input(
        placeholder="What is your question on the document?"
    ):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(
                st.container(), msgs, calculate_similarity=flag_similarity_out
            )
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(
                user_query, callbacks=[retrieval_handler, stream_handler]
            )

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:

        def clear_chat_history():
            msgs.clear()
            msgs.add_ai_message(
                "Welcome to actuarial document summarizer and Q&A tool!"
            )

        st.button(
            label="Clear history",
            use_container_width=True,
            on_click=clear_chat_history,
            help="Retrievals use your conversation history, which will influence future outcomes. Clear history to start fresh on a new topic.",
        )
    with col2:

        def convert_df(msgs):
            df = []
            for msg in msgs.messages:
                df.append({"type": msg.type, "content": msg.content})

            df = pd.DataFrame(df)
            return df.to_csv().encode("utf-8")

        st.download_button(
            label="Download history",
            help="Download chat history in CSV",
            data=convert_df(msgs),
            file_name="chat_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    link = "https://github.com/DanTCIM/doc-summary-qna"
    st.caption(
        f"🖋️ The Python code and documentation of the project are in [GitHub]({link})."
    )
