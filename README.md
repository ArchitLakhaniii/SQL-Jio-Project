# SQL Chatbot with AI-Powered Query Generation

## Overview

This project is a web-based SQL Chatbot application built using Streamlit. The chatbot uses a Large Language Model (LLM) to generate SQL queries from natural language questions, enabling users to interact with a MySQL database without needing to write SQL code themselves. The LLM is powered by the `Ollama` model, and the application is enhanced with various Python libraries like `LangChain`, `pygwalker`, and more.

## Features

- **Natural Language Processing (NLP)**: Converts user questions into SQL queries.
- **Dynamic Interaction**: Users can ask questions in natural language, and the chatbot will attempt to generate and execute the corresponding SQL query.
- **Database Integration**: Connects to a MySQL database and retrieves data based on the generated SQL queries.
- **Interactive Data Display**: The results of the SQL queries are displayed interactively using pandas DataFrames.
- **Automatic Table & Column Recognition**: The application extracts metadata from the MySQL database schema to help the LLM generate accurate SQL queries.
- **Embeddings for Similarity Matching**: Uses embeddings to find the most relevant table or column information based on the user's question.

## Technologies Used

- **Streamlit**: For building the web application.
- **LangChain**: Provides utilities and models for interacting with LLMs.
- **Ollama**: A community-provided LLM used to process natural language inputs and generate SQL queries.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **Pandas**: For handling and displaying tabular data.
- **MySQL**: The relational database used for storing and querying data.
- **PyGWalker**: For interactive visualization of the data.
- **Inflect**: A library for converting numbers to words (though not directly used in the code).

## How It Works

### 1. **Streamlit App Initialization**
   - The application initializes with Streamlit's `set_page_config()` to set the page title and layout.
   - Session state is used to store user messages and query results across interactions.

### 2. **Database Connection**
   - The app connects to a MySQL database using `SQLDatabase` from the `LangChain` library. The database name is `Chinook`, and connection details are provided in the `db_uri`.

### 3. **Handling User Input**
   - The user's natural language question is captured using `st.chat_input()`.
   - This question is then processed to generate an appropriate SQL query.

### 4. **Generating SQL Queries with LLM**
   - The `LLM()` function is responsible for generating and executing SQL queries.
   - The function retrieves detailed information about the database schema (tables and columns) using `get_columns_info()`.
   - Using the `OllamaEmbeddings`, the user's question is matched to the most relevant table or column information.
   - A prompt is created using `ChatPromptTemplate`, which includes the matched data, and the format instructions are defined by `StructuredOutputParser`.
   - The LLM processes the prompt and attempts to generate a SQL query.
   - The generated query is executed, and the results are displayed back to the user.

### 5. **Handling SQL Query Execution and Output**
   - The output of the SQL query is displayed using pandas DataFrames. If no data is returned or if an error occurs, the application handles these cases gracefully by notifying the user.

### 6. **Embedding and Similarity Matching**
   - The application uses `OllamaEmbeddings` to embed the database schema descriptions and user question to find the most relevant schema information using cosine similarity.

## Files Description

### 1. **main.py**

This file contains the main logic of the Streamlit application, including user interaction, database connection, and query processing.

- **`main()`**: The entry point of the application. Sets up the page configuration, initializes session states, and handles user input.
- **`LLM()`**: Processes the user's natural language question, generates a SQL query using the LLM, and executes the query on the MySQL database.
- **`get_columns_info()`**: Connects to the MySQL database and retrieves metadata (table and column information) to assist the LLM in query generation.
- **`OllamaEmbedder()`**: Embeds the database schema descriptions and user question to find the most relevant schema information using cosine similarity.

### 2. **streamlit.py**

This file is a stub for the Streamlit application that imports the main logic and runs the app. It sets up the Streamlit environment and calls the necessary functions from `main.py` to build the chatbot interface.

## Getting Started

### Prerequisites

- Python 3.x
- MySQL database with the `Chinook` schema
- The following Python packages:
  - `streamlit`
  - `pandas`
  - `langchain_community`
  - `pygwalker`
  - `mysql-connector-python`
  - `sklearn`
  - `inflect`

### Installation

1. Clone the repository.
2. Install the required packages using `pip`:
   ```bash
   pip install streamlit pandas mysql-connector-python scikit-learn inflect
   pip install git+https://github.com/langchain-ai/langchain.git
   pip install pygwalker
