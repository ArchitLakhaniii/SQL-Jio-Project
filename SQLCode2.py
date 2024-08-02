import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
import json
import re
import inflect
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import pygwalker as pyg
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Initialize the inflect engine for number to words conversion (not used in this code)
    p = inflect.engine()

    # Set the configuration of the Streamlit page
    st.set_page_config(
        page_title="SQL ChatBot",
        layout="wide"
    )
    st.title("SQL Chatbot")  # Set the title of the app

    # Initialize session states to store messages and DataFrame
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "df" not in st.session_state:
        st.session_state.df = None

    # Define database connection details and connect to the MySQL database
    db_name = "Chinook"
    db_uri = f"mysql+mysqlconnector://root:Jyoti1974!@localhost:3306/{db_name}"
    db = SQLDatabase.from_uri(db_uri)

    # Initialize the large language model (LLM) with the specified model
    llm = Ollama(model="codeqwen")

    # Display past messages in the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get the user's input question from the chat input box
    user_question = st.chat_input("Ask me Anything")

    # Call the function to process the user's question and generate a response
    LLM(
        user_question=user_question,
        llm=llm,
        db_name=db_name,
        db=db
    )

def LLM(user_question, llm, db_name, db):
    # Check if there is a user question to process
    if user_question:
        # Display the user's question in the chat and store it in session state
        with st.chat_message("user"):
            st.write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

        # Show a spinner while the AI processes the query
        with st.spinner("Thinking..."):
            # Retrieve the column and table information from the database
            documents = get_columns_info(schema=db_name)
            # Find the most relevant table information based on the user's question
            data = OllamaEmbedder(documents=documents, user_question=user_question)
            st.write(data)

            # Set up the response schema and output parser for the LLM
            sql_query_schema = ResponseSchema(name="SQL Query", description="This is the SQL Query to pass to the database")
            response_schemas = [sql_query_schema]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            st.write(format_instructions)

            # Define the prompt template for generating SQL queries
            template = """
            You are an advanced AI designed to generate precise SQL queries from natural language questions based on detailed information about a MySQL database.

            **Your task is to:**
            1. Understand the Table Names: Identify and use the correct table names. Accurate table names are crucial for forming the correct query.
            2. Analyze Table Descriptions: Utilize the detailed descriptions of each table to correctly understand the structure and relationships of the data.
            3. Refer to Table Details: Use the specific details from each table to create accurate SQL queries. This includes understanding the data and its format.
            4. Generate the SQL Query: Formulate a SQL query that retrieves the relevant data based on the user's question and provided database information.
            5. Ensure the query:
            - Orders the results by relevant columns when applicable.
            - Selects only the necessary columns relevant to the user's question.
            - Is syntactically correct and logically sound.
            6. Do not make any data modification statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.
            7. If the user's question does not relate to the database, respond with "I don't know."

            **Provided Information**:
            {data}

            **Format Instructions**:
            {format_instructions}
            """

            # Format the prompt with the table information and format instructions
            prompt = ChatPromptTemplate.from_template(template=template)
            messages = prompt.format_messages(
                data=data, format_instructions=format_instructions
            )
            # Get the response from the LLM
            response = llm.invoke(messages[0].content)
            st.write("LLM Response:", response)
            
            # Extract the JSON-formatted SQL query from the LLM response
            json_pattern = r'\{(?:[^{}"]|"[^"]*"|\d+|true|false|null)*\}'
            match = re.search(json_pattern, response)
            if match:
                json_string = match.group(0)
                try:
                    # Parse the SQL query and execute it on the database
                    query_dict = json.loads(json_string)
                    sql_query = query_dict.get("SQL Query", "")
                    output = db._execute(sql_query)
                    st.write(output)
                    
                    if output:
                        # Convert the query output to a DataFrame and display it
                        df = pd.DataFrame(output)
                        st.session_state.df = df
                        
                        with st.chat_message("assistant"):
                            st.write("DataFrame:", df)
                        st.session_state.messages.append({"role": "assistant", "content": df.to_string()})
                    else:
                        # Handle case where no data is returned from the query
                        error = "No data returned from the query."
                        st.write(error)
                        st.session_state.messages.append({"role": "assistant", "content": error})
                except json.JSONDecodeError as e:
                    # Handle JSON decoding errors and recursively retry processing the question
                    error = f"Error decoding JSON: {e}"
                    LLM(
                        user_question=user_question,
                        llm=llm,
                        db_name=db_name,
                        db=db
                    )
                except Exception as e:
                    # Handle SQL query execution errors and recursively retry processing the question
                    error = f"Error executing SQL query: {e}"
                    LLM(
                        user_question=user_question,
                        llm=llm,
                        db_name=db_name,
                        db=db
                    )
            else:
                # Handle case where no valid JSON string is found in the LLM response
                error = "No JSON string found."
                LLM(
                    user_question=user_question,
                    llm=llm,
                    db_name=db_name,
                    db=db
                )

def get_columns_info(schema):
    # Connect to the MySQL database and retrieve information about tables and columns
    import mysql.connector
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Jyoti1974!',
        database=schema
    )
    cursor = conn.cursor()
    query = f"""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = '{schema}';
    """
  
    cursor.execute(query)
    columns_info = cursor.fetchall()

    conn.close()

    # Format the retrieved information into a dictionary for easy access
    formatted_info = {}
    
    for row in columns_info:
        table_name, column_name, data_type, is_nullable = row

        if table_name not in formatted_info:
            formatted_info[table_name] = []

        formatted_info[table_name].append(
            f"{column_name} (Data Type: {data_type}, Nullable: {is_nullable})"
        )

    description_list = []
    
    # Create detailed descriptions of each table and its columns, including example queries
    for table_name, columns in formatted_info.items():
        columns_description = "This table has the following columns: "
        columns_description += ", ".join(
            [f"the first one is {columns[0]}" if i == 0 else f"next is {col}" for i, col in enumerate(columns)]
        )
        
        examples = []
        
        # Add example queries based on table and column names
        examples.append({
            "input": f"Give me all the data from {table_name}",
            "query": f"SELECT * FROM {table_name};"
        })
        
        for column in columns:
            column_name = column.split(" ")[0]  
            examples.append({
                "input": f"Give me the information from {column_name} in {table_name}",
                "query": f"SELECT {column_name} FROM {table_name};"
            })
        
        if len(columns) > 1:
            examples.append({
                "input": f"Give me all data where {columns[1].split(' ')[0]} is not null in {table_name}",
                "query": f"SELECT * FROM {table_name} WHERE {columns[1].split(' ')[0]} IS NOT NULL;"
            })

        if len(columns) > 2:
            examples.append({
                "input": f"Get the count of {columns[2].split(' ')[0]} grouped by {columns[1].split(' ')[0]} in {table_name}",
                "query": f"SELECT {columns[1].split(' ')[0]}, COUNT({columns[2].split(' ')[0]}) FROM {table_name} GROUP BY {columns[1].split(' ')[0]};"
            })

        if len(columns) > 3:
            examples.append({
                "input": f"Get the distinct values of {columns[3].split(' ')[0]} from {table_name}",
                "query": f"SELECT DISTINCT {columns[3].split(' ')[0]} FROM {table_name};"
            })
        
        if len(columns) > 4:
            examples.append({
                "input": f"Get the average of {columns[4].split(' ')[0]} grouped by {columns[1].split(' ')[0]} in {table_name}",
                "query": f"SELECT {columns[1].split(' ')[0]}, AVG({columns[4].split(' ')[0]}) FROM {table_name} GROUP BY {columns[1].split(' ')[0]};"
            })
        
        if len(columns) > 5:
            examples.append({
                "input": f"Get the maximum value of {columns[5].split(' ')[0]} from {table_name}",
                "query": f"SELECT MAX({columns[5].split(' ')[0]}) FROM {table_name};"
            })
        
        examples = examples[:20]  # Limit the number of examples

        examples_text = "Here are some example queries:\n" + "\n".join(
            [f"Input: {example['input']}\nQuery: {example['query']}" for example in examples]
        )
        
        table_description = table_name + ": " + columns_description + ". " + examples_text

        description_list.append(table_description)

    return description_list

def OllamaEmbedder(documents, user_question):
    # Embed the documents (table descriptions) and the user's question using OllamaEmbeddings
    ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")
    doc_embeddings = ollama_emb.embed_documents(documents)
    query_embedding = ollama_emb.embed_query(user_question)

    # Compute cosine similarity between the question and each document
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    most_similar_idx = np.argmax(similarities)

    # Return the most similar document based on the user's question
    data = documents[most_similar_idx]
    return data

# Run the app
if __name__ == "__main__":
    main()
