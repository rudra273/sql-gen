import json
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
import os

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_env_variables():
    """Load environment variables"""
    load_dotenv()
    return {
        'azure_endpoint': os.getenv("AZURE_OPENAI_ENDPOINT"),
        'api_key': os.getenv("AZURE_OPENAI_API_KEY"),
        'api_version': "2024-02-01",
        'deployment_name': "text-embedding-ada-002"
    }

def load_json_file(file_path: str) -> Dict:
    """Load any JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in file: {file_path}")
        return {}

def create_combined_vector_store(
    schema_path: str,
    metadata_path: Optional[str] = None,
    persist_dir: str = "vector_store"
) -> None:
    """Create and persist a single vector store containing both schema and metadata."""
    
    # Initialize Azure OpenAI embeddings
    env_vars = load_env_variables()
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=env_vars['azure_endpoint'],
        api_key=env_vars['api_key'],
        api_version=env_vars['api_version'],
        deployment=env_vars['deployment_name'],
        chunk_size=1
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Load schema and metadata
    schema = load_json_file(schema_path)
    metadata = load_json_file(metadata_path) if metadata_path else {}
    
    # Combined lists for texts and metadatas
    all_texts = []
    all_metadatas = []
    
    # Process schema
    schema_text = json.dumps(schema, indent=2)
    schema_chunks = text_splitter.split_text(schema_text)
    
    for chunk in schema_chunks:
        all_texts.append(chunk)
        all_metadatas.append({
            "doc_type": "schema"
        })
    
    # Process metadata if available
    if metadata:
        metadata_text = json.dumps(metadata, indent=2)
        metadata_chunks = text_splitter.split_text(metadata_text)
        
        for chunk in metadata_chunks:
            all_texts.append(chunk)
            all_metadatas.append({
                "doc_type": "metadata"
            })
    
    # Create combined vector store if we have any texts
    if all_texts:
        vector_store = Chroma.from_texts(
            texts=all_texts,
            metadatas=all_metadatas,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vector_store.persist()
        print(f"Combined vector store created in {persist_dir}")
    else:
        print("No valid data to create vector store")

if __name__ == "__main__":
    schema_path = "schema/schema.json"
    metadata_path = "metadata/metadata.json"  # Optional
    
    create_combined_vector_store(
        schema_path=schema_path,
        metadata_path=metadata_path
    )