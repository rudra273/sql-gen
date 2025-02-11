import psycopg2
from typing import Tuple, List, Optional, Dict
import gradio as gr
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

def load_env_variables():
    """Load environment variables"""
    load_dotenv()
    return {
        'azure_endpoint': os.getenv("AZURE_OPENAI_ENDPOINT"),
        'api_key': os.getenv("AZURE_OPENAI_API_KEY"),
        'api_version': "2024-02-01",
        'chat_deployment': os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # for GPT-4
        'embedding_deployment': 'text-embedding-ada-002',
        'db_name': os.getenv("POSTGRES_DB"),
        'db_user': os.getenv("POSTGRES_USER"),
        'db_password': os.getenv("POSTGRES_PASSWORD"),
        'db_host': os.getenv("POSTGRES_HOST"),
        'db_port': os.getenv("POSTGRES_PORT")
    }

def get_relevant_info(
    query: str,
    vector_store_path: str,
    embeddings: AzureOpenAIEmbeddings,
    num_results: int = 5
) -> Tuple[str, str]:
    """Retrieve relevant schema and metadata"""
    try:
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )
        
        # Get schema information
        schema_results = vector_store.similarity_search(
            query,
            k=num_results,
            filter={"doc_type": "schema"}
        )
        schema_context = "\n\n".join(doc.page_content for doc in schema_results)
        
        # Get metadata information
        metadata_results = vector_store.similarity_search(
            query,
            k=num_results,
            filter={"doc_type": "metadata"}
        )
        metadata_context = "\n\n".join(doc.page_content for doc in metadata_results)
        
        return schema_context, metadata_context
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return "", ""

def generate_sql_query(
    user_query: str,
    schema_context: str,
    metadata_context: str,
    llm: AzureChatOpenAI
) -> str:
    """Generate SQL query using context"""
    if not schema_context:
        return "Error: No schema information available."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert PostgreSQL query generator. Generate queries based on the provided schema and context.

Rules:
1. Use ONLY tables and columns from the schema
2. Follow the exact schema for names
3. Apply business rules from metadata
4. Ensure proper JOIN conditions
5. Handle NULL values appropriately
6. Return ONLY the SQL query without explanation"""),
        ("human", """Schema Information:
{schema}

Additional Context:
{context}

User Query: {query}

Generate the SQL query:""")
    ])
    
    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(
            schema=schema_context,
            context=metadata_context,
            query=user_query
        )
        
        return response.replace("```sql", "").replace("```", "").strip()
    except Exception as e:
        return f"Error generating query: {str(e)}"

def execute_sql_query(
    query: str,
    env_vars: Dict
) -> Tuple[List, Optional[List[str]], Optional[str]]:
    """Execute SQL query against PostgreSQL"""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=env_vars['db_name'],
            user=env_vars['db_user'],
            password=env_vars['db_password'],
            host=env_vars['db_host'],
            port=env_vars['db_port']
        )
        cursor = conn.cursor()
        
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        return results, column_names, None
        
    except Exception as e:
        return [], None, str(e)
    finally:
        if conn:
            conn.close()

def format_results(results: List, columns: Optional[List[str]], error: Optional[str]) -> str:
    """Format query results for display"""
    if error:
        return f"Error: {error}"
    
    if not results:
        return "No results found."
    
    output = []
    
    if columns:
        output.append(" | ".join(columns))
        output.append("-" * len(output[0]))
    
    for row in results[:50]:
        output.append(" | ".join(str(value) for value in row))
    
    if len(results) > 50:
        output.append(f"\n... and {len(results) - 50} more rows")
    
    return "\n".join(output)

def process_query(
    user_query: str,
    chat_history: List[Tuple[str, str]],
    vector_store_path: str,
    env_vars: Dict
) -> Tuple[str, List[Tuple[str, str]]]:
    """Process user query and update chat history"""
    try:
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=env_vars['azure_endpoint'],
            api_key=env_vars['api_key'],
            api_version=env_vars['api_version'],
            azure_deployment=env_vars['embedding_deployment']
        )
        
        # Initialize LLM
        llm = AzureChatOpenAI(
            azure_endpoint=env_vars['azure_endpoint'],
            api_key=env_vars['api_key'],
            api_version=env_vars['api_version'],
            azure_deployment=env_vars['chat_deployment'],
            temperature=0
        )
        
        # Get relevant information
        schema_context, metadata_context = get_relevant_info(
            user_query,
            vector_store_path,
            embeddings
        )
        
        if not schema_context:
            response = "Error: Could not retrieve schema information."
            chat_history.append((user_query, response))
            return "", chat_history
        
        # Generate SQL
        sql_query = generate_sql_query(
            user_query,
            schema_context,
            metadata_context,
            llm
        )
        
        if sql_query.startswith("Error"):
            chat_history.append((user_query, sql_query))
            return "", chat_history
        
        # Execute query
        results, columns, error = execute_sql_query(sql_query, env_vars)
        
        # Format response
        response = (
            f"Generated SQL Query:\n```sql\n{sql_query}\n```\n\n"
            f"Results:\n{format_results(results, columns, error)}"
        )
        
        # Update history
        chat_history.append((user_query, response))
        return "", chat_history
        
    except Exception as e:
        error_response = f"An error occurred: {str(e)}"
        chat_history.append((user_query, error_response))
        return "", chat_history

def create_gradio_interface(
    vector_store_path: str,
    env_vars: Dict
):
    """Create Gradio interface for SQL query generation"""
    
    with gr.Blocks() as interface:
        gr.Markdown("# Natural Language to PostgreSQL Query Generator")
        # gr.Markdown("""
        # Ask questions about your PostgreSQL database in natural language.
        # The system will:
        # 1. Generate appropriate SQL queries using your database schema
        # 2. Execute them against your PostgreSQL database
        # 3. Show you both the query and results
        
        # Note: The system uses schema information and metadata to generate accurate queries.
        # """)
        
        chatbot = gr.Chatbot(
            label="Chat History",
            height=500
        )
        
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Example: Show me all orders from last month with their customer details",
            lines=3
        )
        
        clear = gr.ClearButton([msg, chatbot])
        
        msg.submit(
            fn=lambda q, h: process_query(
                user_query=q,
                chat_history=h,
                vector_store_path=vector_store_path,
                env_vars=env_vars
            ),
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
    
    return interface

def main():
    # Load environment variables
    env_vars = load_env_variables()
    
    # Create and launch interface
    demo = create_gradio_interface(
        vector_store_path="vector_store",
        env_vars=env_vars
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()