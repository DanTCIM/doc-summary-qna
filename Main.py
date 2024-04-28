import os
import re
import streamlit as st

from common.utils import (
    StreamHandler,
    PrintRetrievalHandler,
    embeddings_model,
)

from common.prompt import summary_prompt
from common.sidebar import sidebar_content
from common.chat_history import display_chat_history, clear_chat_history, convert_df

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import Docx2txtLoader
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getvalue())
    if uploaded_file.name.endswith(".pdf"):
        return PDFMinerLoader(uploaded_file.name, concatenate_pages=True)
    elif uploaded_file.name.endswith(".docx"):
        return Docx2txtLoader(uploaded_file.name)
    elif uploaded_file.name.endswith(".txt"):
        return TextLoader(uploaded_file.name)
    else:
        raise ValueError(
            "Unsupported file format. Please upload a file in a supported format."
        )


def text_split_fn(loaded_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    return text_splitter.split_documents(loaded_doc)


def setup_llm(use_anthropic, model_name):
    if use_anthropic:
        return ChatAnthropic(
            model_name=model_name,
            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"],
            temperature=0,
            streaming=True,
        )
    else:
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            temperature=0,
            streaming=True,
        )


def setup_qa_chain(llm, retriever, memory):
    return ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )


def main():
    # API Key Setup
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

    # Initialize Pinecone connection
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

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

    # Initialize session state variables
    if "curr_file" not in st.session_state:
        st.session_state.curr_file = None

    if "prev_file" not in st.session_state:
        st.session_state.prev_file = None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "loaded_doc" not in st.session_state:
        st.session_state.loaded_doc = None

    if uploaded_file is not None:
        st.session_state.curr_file = uploaded_file.name

    if st.session_state.curr_file != st.session_state.prev_file:
        with st.spinner("Extracting text and converting to embeddings..."):
            loader = load_document(uploaded_file)
            st.session_state.loaded_doc = loader.load()

            splits = text_split_fn(st.session_state.loaded_doc)

            namespace = re.sub(r"[^a-zA-Z0-9 \n\.]", "_", st.session_state.curr_file)

            st.session_state.vectorstore = PineconeVectorStore(
                index=pc.Index("streamlit"),
                embedding=embeddings_model,
                namespace=namespace,
            )
            index = pc.Index("streamlit")
            try:
                index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                pass

            st.session_state.vectorstore.add_documents(
                documents=splits,
                namespace=namespace,
            )
            try:
                os.remove(uploaded_file.name)
            except Exception as e:
                pass

        st.session_state.prev_file = st.session_state.curr_file

    # LLM flag for augmented generation (the flag only applied to llm, not embedding model)
    USE_Anthropic = True

    if USE_Anthropic:
        model_name = "claude-3-sonnet-20240229"
    else:
        # model_name = "gpt-3.5-turbo"
        model_name = "gpt-4-turbo"  # gpt-4 seems to be slow

    # Sidebar
    sidebar_content(model_name)

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
    display_chat_history(msgs)

    if st.session_state.vectorstore is not None:

        # Retrieve and RAG chain
        # Create a retriever using the vector database as the search source
        search_kwargs = {"k": st.session_state.num_source}

        if st.session_state.flag_mmr:
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    **search_kwargs,
                    "lambda_mult": st.session_state._lambda_mult,
                },
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
        llm = setup_llm(USE_Anthropic, model_name)

        # Define LLM chain to use for summarizer
        summary_llm_chain = LLMChain(
            llm=llm, prompt=PromptTemplate.from_template(summary_prompt)
        )
        # Define StuffDocumentsChain for summary creation
        stuff_chain = StuffDocumentsChain(
            llm_chain=summary_llm_chain, document_variable_name="text"
        )

        # Define Q&A chain
        qa_chain = setup_qa_chain(llm, retriever, memory)

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
                    st.container(),
                    msgs,
                    calculate_similarity=st.session_state.flag_similarity_out,
                )
                stream_handler = StreamHandler(st.empty())
                response = qa_chain.run(
                    user_query, callbacks=[retrieval_handler, stream_handler]
                )

    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                label="Clear history",
                use_container_width=True,
                on_click=lambda: clear_chat_history(msgs),
                help="Retrievals use your conversation history, which will influence future outcomes. Clear history to start fresh on a new topic.",
            )
        with col2:
            st.download_button(
                label="Download history",
                help="Download chat history in CSV",
                data=convert_df(msgs),
                file_name="chat_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if st.button(
            "Clear Data",
            help="Clears uploaded data, vectorstore, and conversation history.",
            use_container_width=True,
        ):
            if "vectorstore" in st.session_state:
                try:
                    index = pc.Index("streamlit")
                    namespace = namespace = re.sub(
                        r"[^a-zA-Z0-9 \n\.]", "_", st.session_state.curr_file
                    )
                    index.delete(delete_all=True, namespace=namespace)
                    st.session_state.clear()
                    # st.session_state.vectorstore = None

                except Exception as e:
                    pass

        link = "https://github.com/DanTCIM/doc-summary-qna"
        st.caption(
            f"🖋️ The Python code and documentation of the project are in [GitHub]({link})."
        )


if __name__ == "__main__":
    main()
