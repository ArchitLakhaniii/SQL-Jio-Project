# Importing necessary libraries and modules
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
    # Initialize inflect engine for pluralization
    p = inflect.engine()

    # Set the configuration for the Streamlit page
    st.set_page_config(
        page_title="SQL ChatBot",
        layout="wide"
    )
    st.title("SQL Chatbot")  # Set the title for the Streamlit app
    
    # Initialize session state variables if they do not exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "df" not in st.session_state:
        st.session_state.df = None

    # Define database name and URI
    db_name = "Chinook"
    db_uri = f"mysql+mysqlconnector://root:Jyoti1974!@localhost:3306/{db_name}"
    db = SQLDatabase.from_uri(db_uri)  # Create an SQLDatabase instance

    # Initialize the language model (LLM) with the "codeqwen" model
    llm = Ollama(model="codeqwen")

    # Display previous chat messages stored in session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input through the chat interface
    user_question = st.chat_input("Ask me Anything")

    # Generate document descriptions and embeddings based on the database schema
    documents = LLMDetail(get_columns_info(schema=db_name))
    data = OllamaEmbedder(documents=documents, user_question=user_question)
    
    # Generate and execute SQL query based on the user's question
    LLM(
        user_question=user_question,
        llm=llm,
        db_name=db_name,
        db=db
    )

def LLM(user_question, llm, db_name, db, data):
    i = 0
    while i < 3:  # Retry loop for error handling, tries 3 times max
        if user_question:
            # Display user's question in the chat interface
            with st.chat_message("user"):
                st.write(user_question)
                st.session_state.messages.append({"role": "user", "content": user_question})

            with st.spinner("Thinking..."):
                # Define the response schema for the SQL query
                sql_query_schema = ResponseSchema(name="SQL Query", description="This is the SQL Query to pass to the database")
                response_schemas = [sql_query_schema]
                output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
                format_instructions = output_parser.get_format_instructions()
                st.write(format_instructions)
                
                # Create the prompt template for the LLM to generate SQL queries
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
                
                # Format the prompt with provided data and format instructions
                prompt = ChatPromptTemplate.from_template(template=template)
                messages = prompt.format_messages(
                    data=data, format_instructions=format_instructions
                )
                
                # Invoke the LLM to generate the SQL query
                response = llm.invoke(messages[0].content)
                st.write("LLM Response:", response)
                
                # Extract the SQL query from the LLM response
                json_pattern = r'\{(?:[^{}"]|"[^"]*"|\d+|true|false|null)*\}'
                match = re.search(json_pattern, response)
                if match:
                    json_string = match.group(0)
                    try:
                        query_dict = json.loads(json_string)
                        sql_query = query_dict.get("SQL Query", "")
                        output = db._execute(sql_query)  # Execute the SQL query
                        st.write(output)
                        
                        if output:
                            df = pd.DataFrame(output)
                            st.session_state.df = df
                            
                            # Display the output DataFrame in the chat interface
                            with st.chat_message("assistant"):
                                st.write("DataFrame:", df)
                            st.session_state.messages.append({"role": "assistant", "content": df.to_string()})
                        else:
                            error = "No data returned from the query."
                            st.write(error)
                            st.session_state.messages.append({"role": "assistant", "content": error})

                    except json.JSONDecodeError as e:
                        error = f"Error decoding JSON: {e}"
                        if i < 3:
                            i += 1
                            continue
                        st.write(error)
                        st.session_state.messages.append({"role": "assistant", "content": error})

                    except Exception as e:
                        error = f"Error executing SQL query: {e}"
                        if i < 3:
                            i += 1
                            continue
                        st.write(error)
                        st.session_state.messages.append({"role": "assistant", "content": error})
                else:
                    error = "No JSON string found."
                    if i < 3:
                        i += 1
                        continue
                    st.write(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})

def get_columns_info(schema):
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

    formatted_info = {}
    
    for row in columns_info:
        table_name, column_name, data_type, is_nullable = row

        if table_name not in formatted_info:
            formatted_info[table_name] = []

        formatted_info[table_name].append(
            f"{column_name} (Data Type: {data_type}, Nullable: {is_nullable})"
        )

    description_list = []
    
    for table_name, columns in formatted_info.items():
        columns_description = "This table has the following columns: "
        columns_description += ", ".join(
            [f"the first one is {columns[0]}" if i == 0 else f"next is {col}" for i, col in enumerate(columns)]
        )
        
        examples = []
        
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
        
        examples = examples[:20]  # Limit the number of examples to 20

        # Create a description for the table and its columns with example queries
        examples_text = "Here are some example queries:\n" + "\n".join(
            [f"Input: {example['input']}\nQuery: {example['query']}" for example in examples]
        )
        
        table_description = table_name + ": " + columns_description + ". " + examples_text

        description_list.append(table_description)

    return description_list

def LLMDetail(document_info: list):
    # Initialize the language model (LLM) with the "llama3" model
    llm = Ollama(model="llama3")

    # Define the response schema for detailed descriptions of each table and column
    sql_query_schema = ResponseSchema(
        name="Each Table and Column Description", 
        description="Provide a comprehensive and detailed description for each column of each table, including the table itself. Ensure that no details are left out, including data type, nullability, and any other relevant metadata."
    )
    
    response_schemas = [sql_query_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    totalresponse = ""

    # Process the document info in chunks to avoid overwhelming the LLM
    chunk_size = 3 

    for i in range(0, len(document_info), chunk_size):
        document_chunk = document_info[i:i+chunk_size]
        
        template = """
        You are a data expert tasked with generating an exhaustive description of each table and its columns.
        Your response must include:
        - A detailed explanation of each table and what it represents.
        - A precise description of each column, including its name, data type, nullability, and any special characteristics.
        - Example SQL queries that demonstrate how to retrieve data from each table and column.
        - Ensure all aspects are covered in depth without omitting any detail.

        Please structure your output according to the following format:

        {format_instructions}

        Here is the schema and column information to base your response on:

        {documents}

        Make sure to describe all tables and columns presented here.
        """

        # Format the prompt with the document chunk and format instructions
        prompt = ChatPromptTemplate.from_template(template=template)
        messages = prompt.format_messages(
            documents=document_chunk, format_instructions=format_instructions
        )
        
        # Invoke the LLM to generate detailed descriptions
        response = llm.invoke(messages[0].content)
        st.write("LLM Response:", response)
        totalresponse += response
    return totalresponse

def OllamaEmbedder(documents, user_question):
    # Initialize the Ollama Embeddings model
    ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Embed the documents and the user's question
    doc_embeddings = ollama_emb.embed_documents(documents)
    query_embedding = ollama_emb.embed_query(user_question)
    
    # Compute the cosine similarity between the query and document embeddings
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Find the most similar document based on cosine similarity
    most_similar_idx = np.argmax(similarities)
    
    # Return the most relevant document based on the user's question
    data = documents[most_similar_idx]
    return data

# Run the main function when the script is executed
if __name__ == "__main__":
    main()