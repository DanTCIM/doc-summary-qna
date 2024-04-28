import streamlit as st

st.set_page_config(page_title="FAQ", page_icon="📖")

st.title("Frequently Asked Questions")

try:
    st.image("./pages/RAG_process.png", channels="BGR")
except Exception as e:
    st.error(f"Failed to load image: {e}")

st.write(
    """
- The overall RAG process is fast and efficient.
- The retrieval of context is based on a vector search (e.g., similarity), which is fast and efficient.
- The LLM API cost is reasonable, as retrieved contexts (not the full contexts) are used as input to the LLM.
- The RAG instruction indicates how the LLM should respond (e.g., only using retrieved contexts to answer questions).
"""
)

try:
    st.image("./pages/summary_process.png", channels="BGR")
except Exception as e:
    st.error(f"Failed to load image: {e}")

st.write(
    """
- The summary process is slow, as the full contexts are input into the LLM.
- The process leverages a large context window of the LLM (e.g., 200k tokens for Claude 3). Any information beyond this window is discarded.
- The LLM API cost is more expensive, as the full contexts are used as input to the LLM.
- The summary instruction provides a format for summarizing the full document.
"""
)

with st.expander("What is RAG?"):
    st.write(
        """
        RAG stands for Retrieval-Augmented Generation. It is a technique used in natural language processing (NLP) that combines the capabilities of a retrieval system and a generative language model. The retrieval system is used to find relevant information from a corpus of documents based on the input query, while the language model is used to generate a response based on the retrieved information and the query.
    """
    )

with st.expander("Is RAG accurate?"):
    st.write(
        """
        RAG is typically more accurate and provides more relevant responses compared to a generic question asked on a language model alone. However, RAG is not perfect due to potential errors in the retrieval process and limitations of the generative model. Additionally, language models may provide definite answers where the initial question may not have a clear or definitive answer. This "false precision" and "false fluency" is a limitation that RAG also shares with language models.
        """
    )

with st.expander(
    "What is the difference between RAG and traditional information retrieval?"
):
    st.write(
        """
        Traditional information retrieval systems typically retrieve relevant documents or passages based on keyword matching or other statistical methods. RAG, on the other hand, combines the retrieval process with a generative language model, allowing for more natural and context-aware responses. The language model can synthesize information from multiple retrieved sources, fill in gaps, and generate coherent and fluent responses.
    """
    )

with st.expander("Explain RAG parameters."):
    st.write(
        "You can adjust the following parameters to customize the retrieval and generation process:"
    )
    st.write(
        "- **Top N Sources to View**: Determines how many of the top retrieved sources/chunks to display in the interface. The search algorithm retrieves 20 chunks in total."
    )
    st.write(
        "- **Diversity Search**: Enabling this allows the retrieved sources to be more diverse, preventing repeated information that is very close to your initial query. This can be useful for getting a broader perspective."
    )
    st.write(
        "- **Output similarity score**: You may output similarity score between initial query and retrieved chunks. Allowing the score output may make the retrieval process slower."
    )


with st.expander("What large language model (LLM) is used?"):
    st.write(
        """
        The application uses Anthropic's Claude 3 model for generating responses. The model is chosen due to its ability to get input size as large as 200 thousand tokens (about 400-500 pages of texts). Claude 3 is also used to develop the summary of each document.
    """
    )

with st.expander("What is embedding?"):
    st.write(
        """
        Embedding is the process of converting text data (words, sentences, documents) into high-dimensional vector representations. These vector representations capture the semantic meaning and context of the text data, allowing for efficient storage, retrieval, and comparison of the text using vector operations.
    """
    )

with st.expander("What embedding model is used?"):
    st.write(
        "This application uses the OpenAI embeddings large model for generating text embeddings."
    )

with st.expander("What is a vector database?"):
    st.write(
        """
        A vector database is a specialized database designed for storing and querying high-dimensional vector representations (embeddings) of data. Unlike traditional databases that store structured data, vector databases are optimized for working with dense vector representations, enabling efficient similarity searches, nearest neighbor queries, and other vector operations.
    """
    )

with st.expander("What vector database is used?"):
    st.write(
        """
        Pinecone vector database is used in this application. While open-source databases such as Chroma can be utilized, Pinecone is chosen for its scalability and serverless features. Pinecone is a managed vector database that allows for efficient storage and retrieval of high-dimensional vectors, making it suitable for applications dealing with large-scale vector data. Its serverless architecture, hosted on AWS, eliminates the need for managing infrastructure, enabling developers to focus on building the application. 
    """
    )


with st.expander(
    "What does the vector database look like? How does it work with retrieval?"
):
    st.write(
        """
        A vector database contains embeddings for each document or chunk of text. The database is organized into separate collections. A query is performed within a specific collection. For each document, the text is split into smaller chunks. Each chunk contains metadata, such as the document name. An embedding model assigns a dense vector representation (embedding) to each chunk. These embeddings are then stored in the corresponding collection. 

        During the retrieval process, the user's query is also embedded using the same embedding model. The query embedding is then compared against the embeddings in the database using a similarity metric like cosine similarity. The chunks or documents with embeddings most similar to the query embedding are retrieved as the most relevant results.
    """
    )


with st.expander(
    "What are considerations when splitting a document into chunks? What method did the application use?"
):
    st.write(
        """
        One can use a fixed number of tokens or pages as a straightforward method for chunking, but the application chose to use an embedding similarity method. This approach aims to create more coherent and semantically meaningful chunks by grouping text based on their embeddings' similarity.

        A fixed number of tokens or pages can be used for chunking, and this can be computationally cheaper. Generally, smaller chunks are preferred for question-answering applications where precise retrieval is crucial, while larger chunks may be better suited for summarization or information extraction tasks that require more context.
    """
    )

with st.expander("What instruction is used to summarize the document?"):
    st.write(
        """
        The following is the system instruction used to summarize an actuarial document.\n\nWrite a summary of the following document using the format provided. The summary includes the title, publisher and published date if available, purpose and scope, key points, conclusion and implications, and key words. The summary should be comprehensive yet brief, aiming for a reading time of no more than one minute. Avoid any translation or substitution of actuarial terms in the document. When starting a summary, begin the summary with "Title:" without saying "Here is a summary". Here is the summary format:

Title: [title of the document]\n\nPublisher and Published Date: [published date and publisher's name]\n\nPurpose and Scope: [note the purpose and scope]\n\nKey Points:\n- [indicate key points in bullet point format]\n\nConclusions and Implications: [describe conclusion and implications for regulatory and practical purposes in the actuarial field]\n\nKey words: [indicate top five key words from the document]
    """
    )
