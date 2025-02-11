import psycopg2
from typing import Tuple, List, Optional, Dict
import gradio as gr
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

def load_env_variables():
    """Load environment variables"""
    load_dotenv()
    return {
        'azure_endpoint': os.getenv("AZURE_OPENAI_ENDPOINT"),
        'api_key': os.getenv("AZURE_OPENAI_API_KEY"),
        'api_version': "2024-02-01",
        'chat_deployment': os.getenv("AZURE_OPENAI_DEPLOYMENT"),
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
        
        schema_results = vector_store.similarity_search(
            query,
            k=num_results,
            filter={"doc_type": "schema"}
        )
        schema_context = "\n\n".join(doc.page_content for doc in schema_results)
        
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
    chat_history: List[Dict[str, str]],
    llm: AzureChatOpenAI
) -> str:
    """Generate SQL query using context and chat history"""
    if not schema_context:
        return "Error: No schema information available."
    
    # Convert chat history to string format
    history_str = "\n".join([
        f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
        for msg in chat_history[-6:]  # Last 3 exchanges (6 messages)
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert PostgreSQL query generator that provides helpful explanations. Generate queries based on the provided schema, context, and chat history.

Rules:
1. Use ONLY tables and columns from the schema
2. Follow the exact schema for names
3. Apply business rules from metadata
4. Ensure proper JOIN conditions
5. Handle NULL values appropriately
6. Ensure string comparisons are case-insensitive using ILIKE
7. First provide a brief explanation of what the query will do
8. Then provide the SQL query

For follow-up questions, use the chat history to maintain context and modify previous queries appropriately."""),
        ("human", """Schema Information:
{schema}

Additional Context:
{context}

Recent Chat History:
{history}

User Query: {query}

Generate explanation and SQL query:""")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "schema": schema_context,
            "context": metadata_context,
            "history": history_str,
            "query": user_query
        })
        
        return response.content
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
    chat_history: List[Dict[str, str]],
    vector_store_path: str,
    env_vars: Dict
) -> Tuple[str, List[Dict[str, str]]]:
    """Process user query and update chat history"""
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=env_vars['azure_endpoint'],
            api_key=env_vars['api_key'],
            api_version=env_vars['api_version'],
            azure_deployment=env_vars['embedding_deployment']
        )
        
        llm = AzureChatOpenAI(
            azure_endpoint=env_vars['azure_endpoint'],
            api_key=env_vars['api_key'],
            api_version=env_vars['api_version'],
            azure_deployment=env_vars['chat_deployment'],
            temperature=0
        )
        
        schema_context, metadata_context = get_relevant_info(
            user_query,
            vector_store_path,
            embeddings
        )
        
        if not schema_context:
            response = "Error: Could not retrieve schema information."
            return "", chat_history + [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": response}
            ]
        
        # Generate response with explanation and SQL
        response = generate_sql_query(
            user_query,
            schema_context,
            metadata_context,
            chat_history,
            llm
        )
        
        # Update history
        return "", chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response}
        ]
        
    except Exception as e:
        error_response = f"An error occurred: {str(e)}"
        return "", chat_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": error_response}
        ]

def execute_query(
    query: str,
    env_vars: Dict
) -> str:
    """Execute the SQL query and return formatted results"""
    results, columns, error = execute_sql_query(query, env_vars)
    return format_results(results, columns, error)

def create_gradio_interface(
    vector_store_path: str,
    env_vars: Dict
):
    """Create Gradio interface with separated query generation and execution"""
    
    with gr.Blocks() as interface:
        gr.Markdown("# Natural Language to PostgreSQL Query Generator")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    type="messages"  # Use new message format
                )
                
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Example: Show me all orders from last month with their customer details",
                    lines=3
                )
                
                clear = gr.ClearButton([msg, chatbot])
            
            with gr.Column(scale=1):
                sql_input = gr.Textbox(
                    label="SQL Query to Execute",
                    placeholder="Paste the generated SQL query here to execute it",
                    lines=5,
                    interactive=True
                )
                
                execute_button = gr.Button("Execute Query")
                
                results_output = gr.Textbox(
                    label="Query Results",
                    lines=10,
                    interactive=False 
                )
        
        # Handle query generation
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
        
        # Handle query execution
        execute_button.click(
            fn=lambda q: execute_query(query=q, env_vars=env_vars),
            inputs=[sql_input],
            outputs=[results_output]
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
    )

if __name__ == "__main__":
    main()