# Actuarial Document Summarizer and Q&A Tool
## 1. Description
This project aims to create a process to help actuaries review actuarial documents. Once a PDF file is uploaded, a user can create a summary or ask questions about the document. The process utilizes the power of a large language model (LLM). The tool is powered by Anthropic's Claude 3 Sonnet.

It is crucial to know that an LLM's response may be inaccurate. Actuaries are strongly encouraged to review the source document after using the tool.

The Retrieval-Augmented Generation (RAG) process used LangChain, a framework for developing applications powered by LLMs.

## 2. Output
### 2.1 Demo App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-summary-qna.streamlit.app/)  
Please visit the Streamlit web app (https://doc-summary-qna.streamlit.app/) to upload your own PDF document. Explore how the tool can help you familiarize yourself with the document and get questions answered about it.

## 3. Model
### 3.1 Conceptual Flow
The tool uses two processes. 

![RAG process](./pages/RAG_process.png)
- The overall RAG process is fast and efficient.
- The retrieval of context is based on a vector search (e.g., similarity), which is fast and efficient.
- The LLM API cost is reasonable, as retrieved contexts (not the full contexts) are used as input to the LLM.
- The RAG instruction indicates how the LLM should respond (e.g., only using retrieved contexts to answer questions).

![Summary process](./pages/summary_process.png)
- The summary process is slow, as the full contexts are input into the LLM.
- The process leverages a large context window of the LLM (e.g., 200k tokens for Claude 3). Any information beyond this window is discarded.
- The LLM API cost is more expensive, as the full contexts are used as input to the LLM.
- The summary instruction provides a format for summarizing the full document.

## 4. Author
Dan Kim 

- [@LinkedIn](https://www.linkedin.com/in/dan-kim-4aaa4b36/)
- dan.kim.actuary@gmail.com (feel free to reach out with questions or comments)

## 5. Date
- Initially published on 4/14/2024
- The contents may be updated from time to time
  
## 6. License
This project is licensed under the Apache License 2.0- see the LICENSE.md file for details.
