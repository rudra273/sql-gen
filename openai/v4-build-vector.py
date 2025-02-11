import sqlite3
import json
from pathlib import Path
from typing import Dict, Optional

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_database_schema(db_path: str) -> Dict:
    """Extract database schema by querying the database directly."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    schema_info = {}
    
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
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        foreign_keys = cursor.fetchall()
        
        schema_info[table] = {
            "columns": [
                {
                    "name": col[1],
                    "type": col[2],
                    "is_primary_key": bool(col[5])
                }
                for col in columns
            ],
            "foreign_keys": [
                {
                    "from_column": fk[3],
                    "to_table": fk[2],
                    "to_column": fk[4]
                }
                for fk in foreign_keys
            ]
        }
    
    conn.close()
    return schema_info

def load_metadata(metadata_path: Optional[str]) -> Dict:
    """Load optional metadata from JSON file."""
    if not metadata_path or not Path(metadata_path).exists():
        return {}
    
    with open(metadata_path, 'r') as f:
        return json.load(f)

def create_vector_stores(
    db_path: str,
    openai_api_key: str,
    metadata_path: Optional[str] = None,
    base_persist_dir: str = "final-vector"
) -> None:
    """Create and persist separate vector stores for schema and metadata."""
    
    # Initialize components
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Create persist directories
    schema_persist_dir = Path(base_persist_dir) / "schema"
    metadata_persist_dir = Path(base_persist_dir) / "metadata"
    
    # Get schema and metadata
    schema = get_database_schema(db_path)
    metadata = load_metadata(metadata_path)
    
    # Process schema information
    schema_texts = []
    schema_metadatas = []
    
    for table_name, table_info in schema.items():
        # Format column information
        column_types = ", ".join(
            f"{col['name']}: {col['type']}"
            for col in table_info['columns']
        )
        
        primary_keys = ", ".join(
            col['name']
            for col in table_info['columns']
            if col['is_primary_key']
        )
        
        # Create schema text
        schema_text = (
            f"Table: {table_name}\n"
            f"Columns: {', '.join(col['name'] for col in table_info['columns'])}\n"
            f"Column Types: {column_types}\n"
            f"Primary Keys: {primary_keys}\n"
            f"Foreign Keys: {json.dumps(table_info['foreign_keys'], indent=2)}"
        )
        schema_texts.append(schema_text)
        schema_metadatas.append({
            "type": "schema",
            "table": table_name,
            "doc_type": "schema"
        })
    
    # Create schema vector store
    schema_store = Chroma.from_texts(
        texts=schema_texts,
        metadatas=schema_metadatas,
        embedding=embeddings,
        persist_directory=str(schema_persist_dir)
    )
    schema_store.persist()
    print(f"Schema vector store created in {schema_persist_dir}")
    
    # Process metadata if available
    if metadata:
        metadata_texts = []
        metadata_metadatas = []
        
        for key, value in metadata.items():
            metadata_text = f"{key}: {json.dumps(value, indent=2)}"
            chunks = text_splitter.split_text(metadata_text)
            
            for chunk in chunks:
                metadata_texts.append(chunk)
                metadata_metadatas.append({
                    "type": "metadata",
                    "key": key,
                    "doc_type": "metadata"
                })
        
        # Create metadata vector store
        metadata_store = Chroma.from_texts(
            texts=metadata_texts,
            metadatas=metadata_metadatas,
            embedding=embeddings,
            persist_directory=str(metadata_persist_dir)
        )
        metadata_store.persist()
        print(f"Metadata vector store created in {metadata_persist_dir}")

if __name__ == "__main__":
    import os
    
    db_path = "sqlite.db"
    metadata_path = "metadata.json"  # Optional
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    create_vector_stores(
        db_path=db_path,
        openai_api_key=openai_api_key,
        metadata_path=metadata_path
    )