import gradio as gr
import sqlite3
import pandas as pd
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import sqlparse
import json

def get_db_schema(db_path: str):
    """Extract database schema""" 
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    schema_info = {}
    unique_values = {}
    
    # Get all tables
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    tables = [table[0] for table in cursor.fetchall()]
    
    for table in tables:
        # Get column information
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        schema_info[table] = {
            "columns": [col[1] for col in columns],
            "types": {col[1]: col[2] for col in columns}
        }
        
        # Get sample values for string columns
        for col in columns:
            if col[2].lower() in ['text', 'varchar', 'char', 'string']:
                try:
                    cursor.execute(f"""
                        SELECT DISTINCT {col[1]} 
                        FROM {table} 
                        WHERE {col[1]} IS NOT NULL 
                        LIMIT 100
                    """)
                    values = [row[0] for row in cursor.fetchall() if row[0]]
                    if values:
                        unique_values[f"{table}.{col[1]}"] = values
                except:
                    continue
    
    conn.close()
    return schema_info, unique_values

def initialize_vector_store(schema_info, examples, embeddings, persist_dir):
    """Initialize vector store with schema and examples"""
    texts = []
    metadatas = []
    
    # Add schema information
    for table, info in schema_info.items():
        schema_text = f"Table: {table}\nColumns: {', '.join(info['columns'])}"
        texts.append(schema_text)
        metadatas.append({"type": "schema", "table": table})
    
    # Add example queries
    for example in examples:
        texts.append(
            f"Input: {example['input']}\n"
            f"SQL: {example['sql']}\n"
            f"Tables: {example['tables']}"
        )
        metadatas.append({"type": "example", "tables": example['tables']})
    
    vector_store = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vector_store.persist()
    return vector_store

def validate_sql(sql: str, db_path: str):
    """Validate SQL query"""
    try:
        sqlparse.parse(sql)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)

def execute_query(query: str, db_path: str):
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return True, df.to_string()
    except Exception as e:
        return False, str(e)

class SQLChatBot:
    def __init__(self, db_path, persist_dir, openai_api_key):
        self.db_path = db_path
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Initialize schema and vector store
        self.schema_info, self.unique_values = get_db_schema(db_path)
        
        example_queries = [
            {
                "input": "Show empty shelves in Walgreens",
                "sql": "SELECT outletname, SHELF, POSITION, PRODUCTNAME FROM customer_observability_360 WHERE outletname LIKE '%Walgreens%' AND ISEMPTY = true",
                "tables": "customer_observability_360"
            }
            # Add more examples here
        ]
        
        self.vector_store = initialize_vector_store(
            self.schema_info, 
            example_queries,
            self.embeddings,
            persist_dir
        )
    
    def generate_sql(self, user_query: str, error_context: str = None):
        """Generate SQL query with context"""
        # Get relevant context
        results = self.vector_store.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        
        error_prompt = f"\nPrevious error: {error_context}" if error_context else ""
        
        prompt = f"""
        Generate a SQL query based on the user's request.
        
        Available Tables and Columns:
        {json.dumps(self.schema_info, indent=2)}
        
        Relevant Context:
        {context}
        
        User Query: {user_query}{error_prompt}
        
        Return response in format:
        QUERY: <sql query>
        TABLES: <tables used>
        EXPLANATION: <explanation>
        """
        
        messages = [
            SystemMessage(content="You are an expert SQL query generator. if the user user input is not releted about createing a queery just be like a simple chatbot and explain what u can do"),
            HumanMessage(content=prompt)
        ]
        print(messages)
        response = self.llm.generate([messages])
        response_text = response.generations[0][0].text
        
        # Check if the response contains the expected format
        if "QUERY:" not in response_text or "TABLES:" not in response_text or "EXPLANATION:" not in response_text:
            return "Error: The response from the model is not in the expected format."
        
        try:
            # Parse response
            sql_query = response_text.split('QUERY:')[1].split('TABLES:')[0].strip()
            tables = response_text.split('TABLES:')[1].split('EXPLANATION:')[0].strip()
            explanation = response_text.split('EXPLANATION:')[1].strip()
            
            return f"QUERY: {sql_query}\nTABLES: {tables}\nEXPLANATION: {explanation}"
        
        except IndexError as e:
            return f"Error parsing the model's response: {str(e)}"
    
    def process_query(self, query: str, error_context: str = None):
        """Process user query and return response"""
        try:
            # Generate SQL
            response = self.generate_sql(query, error_context)
            
            # Parse response
            sql_query = response.split('QUERY:')[1].split('TABLES:')[0].strip()
            tables = response.split('TABLES:')[1].split('EXPLANATION:')[0].strip()
            explanation = response.split('EXPLANATION:')[1].strip()
            
            # Validate SQL
            is_valid, error = validate_sql(sql_query, self.db_path)
            if not is_valid:
                return f"SQL Error: {error}\n\nGenerated query was:\n{sql_query}\n\nPlease rephrase your question or provide more details."
            
            # Execute query
            success, result = execute_query(sql_query, self.db_path)
            if not success:
                return f"Execution Error: {result}\n\nGenerated query was:\n{sql_query}"
            
            return f"""Generated SQL Query:
            {sql_query}

            Tables Used: {tables}

            Explanation: {explanation}

            Results:
            {result}"""
        
        except Exception as e:
            return f"Error: {str(e)}"

def create_chat_interface():
    # Initialize chatbot
    db_path = "sqlite.db"
    persist_dir = "vector_store"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    chatbot = SQLChatBot(db_path, persist_dir, openai_api_key)
    
    # Create interface
    with gr.Blocks() as interface:
        gr.Markdown("# SQL Query Generator")
        gr.Markdown(f"Available tables: {', '.join(chatbot.schema_info.keys())}")
        
        chatbot_component = gr.Chatbot()
        msg = gr.Textbox(label="Ask a question about your data")
        clear = gr.ClearButton([msg, chatbot_component])
        
        def respond(message, history):
            error_context = None
            if history and "Error" in history[-1][1]:
                error_context = history[-1][1]
                
            bot_message = chatbot.process_query(message, error_context)
            history.append((message, bot_message))
            return "", history
        
        msg.submit(respond, [msg, chatbot_component], [msg, chatbot_component])
    
    return interface

# Launch the interface
if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch()