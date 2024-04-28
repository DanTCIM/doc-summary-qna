import streamlit as st


def sidebar_content(model_name):
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
            st.session_state.num_source = st.slider(
                "Top N sources to view:", min_value=4, max_value=20, value=5, step=1
            )
            st.session_state.flag_mmr = st.toggle(
                "Diversity search",
                value=True,
                help="Diversity search, i.e., Maximal Marginal Relevance (MMR) tries to reduce redundancy of fetched documents and increase diversity. 0 being the most diverse, 1 being the least diverse. 0.5 is a balanced state.",
            )
            st.session_state._lambda_mult = st.slider(
                "Diversity parameter (lambda):",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.25,
            )
            st.session_state.flag_similarity_out = st.toggle(
                "Output similarity score",
                value=False,
                help="The retrieval process may become slower due to the cosine similarity calculations. A similarity score of 100% indicates the highest level of similarity between the query and the retrieved chunk.",
            )
