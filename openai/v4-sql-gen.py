import sqlite3
from typing import Tuple, List, Optional, Dict
import gradio as gr
import os
from pathlib import Path
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

def get_relevant_info(
    query: str,
    base_vector_store_path: str,
    embeddings: OpenAIEmbeddings,
    num_results: int = 5
) -> Tuple[str, str]:
    """Retrieve relevant schema and metadata separately."""
    # Load schema vector store
    schema_store = Chroma(
        persist_directory=str(Path(base_vector_store_path) / "schema"),
        embedding_function=embeddings
    )
    
    # Load metadata vector store
    metadata_store = Chroma(
        persist_directory=str(Path(base_vector_store_path) / "metadata"),
        embedding_function=embeddings
    )
    
    # Get relevant schema information
    schema_results = schema_store.similarity_search(
        query,
        k=num_results,
        filter={"doc_type": "schema"}
    )
    schema_context = "\n\n".join(doc.page_content for doc in schema_results)
    
    # Get relevant metadata information
    metadata_results = metadata_store.similarity_search(
        query,
        k=num_results,
        filter={"doc_type": "metadata"}
    )
    metadata_context = "\n\n".join(doc.page_content for doc in metadata_results)

    # print(schema_context, metadata_context)
    
    return schema_context, metadata_context

def generate_sql_query(
    user_query: str,
    schema_context: str,
    metadata_context: str,
    llm: ChatOpenAI
) -> str:
    """Generate SQL query using separate schema and metadata context."""
    
    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert SQL query generator. Generate SQL queries based on the provided database schema and additional context.

Rules:
1. Use ONLY tables and columns mentioned in the schema
2. Use the schema for all table and column names
3. Use metadata context for additional business rules and constraints
4. Ensure proper JOIN conditions based on schema relationships
5. Handle NULL values properly
6. Return ONLY the SQL query without any explanation"""),
        ("human", """Database Schema:
{schema}

Additional Context:
{context}

User Query: {query}

Generate the SQL query:""")
    ])
    
    # Create and run chain
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(
        schema=schema_context,
        context=metadata_context,
        query=user_query
    )

    return response.replace("```sql", "").replace("```", "").strip()
    
    # Clean up response
    # return response.strip().strip('`').strip()

def execute_sql_query(
    query: str,
    db_path: str
) -> Tuple[List, Optional[List[str]], Optional[str]]:
    """Execute SQL query and return results with column names."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        conn.close()
        return results, column_names, None
        
    except Exception as e:
        return [], None, str(e)

def format_results(results: List, columns: Optional[List[str]], error: Optional[str]) -> str:
    """Format query results for display."""
    if error:
        return f"Error: {error}"
    
    if not results:
        return "No results found."
    
    output = []
    
    # Add header
    if columns:
        output.append(" | ".join(columns))
        output.append("-" * len(output[0]))
    
    # Add rows (limit to 50 for display)
    for row in results[:50]:
        output.append(" | ".join(str(value) for value in row))
    
    if len(results) > 50:
        output.append(f"\n... and {len(results) - 50} more rows")
    
    return "\n".join(output)

def process_query(
    user_query: str,
    chat_history: List[Tuple[str, str]],
    db_path: str,
    vector_store_path: str,
    openai_api_key: str
) -> Tuple[str, List[Tuple[str, str]]]:
    """Process user query and update chat history."""
    try:
        # Initialize components
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Get relevant schema and metadata
        schema_context, metadata_context = get_relevant_info(
            user_query,
            vector_store_path,
            embeddings
        )
        
        # Generate SQL
        sql_query = generate_sql_query(
            user_query,
            schema_context,
            metadata_context,
            llm
        )
        
        # Execute query
        results, columns, error = execute_sql_query(sql_query, db_path)
        
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
    db_path: str,
    vector_store_path: str,
    openai_api_key: str
):
    """Create Gradio interface for SQL query generation."""
    
    with gr.Blocks() as interface:
        gr.Markdown("# Natural Language to SQL Query Generator")
        gr.Markdown("""
        Ask questions about your data in natural language.
        The system will:
        - Generate appropriate SQL queries based on schema and metadata
        - Execute them against your database
        - Show you both the query and results
        """)
        
        chatbot = gr.Chatbot(
            label="Chat History",
            height=400
        )
        
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Example: Show me all users who made purchases last month",
            lines=2
        )
        
        clear = gr.ClearButton([msg, chatbot])
        
        msg.submit(
            fn=lambda q, h: process_query(
                user_query=q,
                chat_history=h,
                db_path=db_path,
                vector_store_path=vector_store_path,
                openai_api_key=openai_api_key
            ),
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
    
    return interface

if __name__ == "__main__":
    # Configuration
    DB_PATH = "sqlite.db"
    VECTOR_STORE_PATH = "final-vector"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Create and launch interface
    demo = create_gradio_interface(
        db_path=DB_PATH,
        vector_store_path=VECTOR_STORE_PATH,
        openai_api_key=OPENAI_API_KEY
    )
    demo.launch()

    