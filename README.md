
# Multi-Tenant Agentic RAG System

This project implements a multi-tenant Agentic RAG (Retrieval-Augmented Generation) system using LangGraph. The system is designed to manage data from multiple distinct folders, treating each folder name as a tenant ID.

## Architecture

The system is built on a StateGraph architecture with the following components:

*   **State Schema:** The state of the graph tracks the following information:
    *   `messages`: A list of messages in the conversation.
    *   `intent`: The user's intent, classified as either `rag` or `general`.
    *   `company_name`: The name of the company (tenant) mentioned in the user's query.

*   **Nodes:** The graph consists of the following nodes:
    *   **Router:** This node uses a Large Language Model (LLM) to classify the user's intent and extract the company name from the query.
    *   **Retriever:** This node executes a vector search on a FAISS vector store. It uses a metadata filter to ensure that only data from the specified tenant is retrieved.
    *   **Generator:** This node uses an LLM to synthesize an answer based on the retrieved documents.
    *   **Chat:** This node uses an LLM to provide a general chat response.

*   **Edges:** The nodes are connected by conditional edges based on the user's intent. If the intent is `rag`, the graph routes to the `Retriever` node. If the intent is `general`, the graph routes to the `Chat` node.

## Workflow

1.  **Ingestion:** The ingestion pipeline iterates through the directory structure, loading documents from each tenant's folder. The documents are split into chunks, and embeddings are created for each chunk. The chunks are then stored in a FAISS vector store, with the parent folder name added as metadata to each chunk.

2.  **Routing:** When a user submits a query, the `Router` node classifies the intent and extracts the company name.

3.  **Retrieval:** If the intent is `rag`, the `Retriever` node performs a similarity search on the vector store, filtering by the extracted company name.

4.  **Generation:** The retrieved documents are passed to the `Generator` node, which synthesizes an answer.

5.  **Chat:** If the intent is `general`, the `Chat` node provides a standard chat response.

## Libraries Used

*   **langchain:** Used for building the RAG pipeline and interacting with the LLM.
*   **langgraph:** Used for building the stateful, multi-agent architecture.
*   **langchain-openai:** Used for interacting with the OpenAI API (in the original version).
*   **langchain-community:** Used for interacting with local models like Ollama and Hugging Face.
*   **faiss-cpu:** Used for creating and storing the vector index.
*   **tiktoken:** Used for tokenizing text for the OpenAI models.
*   **pypdf:** Used for loading PDF documents.
*   **sentence-transformers:** Used for creating local embeddings.
*   **ollama:** Used for running local LLMs.

## Tools Used

*   **Ollama:** Used for running the `phi3:latest` model locally.
*   **Hugging Face:** Used for downloading the `all-MiniLM-L6-v2` sentence transformer model.

## How to Run

1.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
2.  **Set up your environment:**
    *   Create a `.env` file in the project root.
    *   Add your OpenAI API key to the `.env` file (if you are using the OpenAI models):
        ```
        OPENAI_API_KEY=your_key_here
        ```
3.  **Run the ingestion script:**
    ```
    python ingestion.py
    ```
4.  **Run the main application:**
    ```
    python main.py "your query"
    ```
