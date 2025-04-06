# Simple Low-Latency Internet RAG Model

This project implements a simple Retrieval-Augmented Generation (RAG) model designed for low-latency responses by leveraging real-time internet search results.

## Key Features

*   **Internet-Connected:** Uses the Brave Search API to fetch up-to-date information from the web, ensuring the model has access to the latest knowledge.
*   **Low Latency:** Powered by the Groq API, known for its high-speed inference, enabling near real-time answers.
*   **Simplicity:** A straightforward implementation focusing on the core RAG pipeline: retrieve relevant web results and generate an answer based on them.

## How it Works

1.  The user provides a query.
2.  The Brave Search API is queried to retrieve relevant web snippets based on the user's query.
3.  These snippets, along with the original query, are passed to a large language model hosted on Groq.
4.  The Groq model synthesizes the information from the web snippets to generate a comprehensive and contextually relevant answer.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shiv207/internet-RAG-model.git
    cd internet-RAG-model
    ```
2.  **Install dependencies:** (Assuming a `requirements.txt` file exists or will be added)
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```plaintext
    GROQ_API_KEY=your_groq_api_key
    BRAVE_API_KEY=your_brave_search_api_key
    ```
4.  **Run the application:**
    ```bash
    python app.py
    ```

## Technologies Used

*   **Groq:** For fast LLM inference.
*   **Brave Search API:** For real-time web search results.
*   **Python:** The primary programming language. 