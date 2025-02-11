import json
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from typing import Dict, Any
from pathlib import Path

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NaN, Infinity, and -Infinity"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return None
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        return super().default(obj)

def read_tabular_file(file_path: str) -> Dict[str, Any]:
    """
    Read various tabular file formats and extract basic metadata about the structure
    Supports: csv, xlsx, xls, tsv, txt (comma or tab separated)
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        
        # Read the file based on extension
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.tsv' or file_extension == '.txt':
            # Try tab first, then comma if that fails
            try:
                df = pd.read_csv(file_path, sep='\t')
            except:
                df = pd.read_csv(file_path, sep=',')
        elif file_extension in ['.xlsx', '.xls']:
            # Read all sheets from Excel
            excel_data = pd.read_excel(file_path, sheet_name=None)
            if len(excel_data) == 1:
                # If only one sheet, use it directly
                df = list(excel_data.values())[0]
            else:
                # If multiple sheets, combine them with sheet name as prefix
                dfs = []
                for sheet_name, sheet_df in excel_data.items():
                    # Add sheet name prefix to columns
                    sheet_df.columns = [f"{sheet_name}_{col}" for col in sheet_df.columns]
                    dfs.append(sheet_df)
                df = pd.concat(dfs, axis=1)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Replace problematic values
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        # Extract basic metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "file_type": file_extension[1:],  # Remove the dot
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": {}
        }
        
        # If it's an Excel file with multiple sheets, add sheet information
        if file_extension in ['.xlsx', '.xls']:
            metadata["sheets"] = list(excel_data.keys())
        
        # Extract column-level metadata
        for column in df.columns:
            non_null_values = df[column].dropna()
            metadata["columns"][column] = {
                "data_type": str(df[column].dtype),
                "null_count": int(df[column].isna().sum()),
                "unique_values": int(non_null_values.nunique()) if len(non_null_values) > 0 else 0,
                "sample_values": non_null_values.head(3).tolist() if len(non_null_values) > 0 else [],
                "min_value": float(non_null_values.min()) if non_null_values.dtype in ['int64', 'float64'] and len(non_null_values) > 0 else None,
                "max_value": float(non_null_values.max()) if non_null_values.dtype in ['int64', 'float64'] and len(non_null_values) > 0 else None
            }
        
        return {
            "data": df.to_dict(orient="records"),
            "metadata": metadata
        }
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def enrich_metadata_with_llm(data: Dict[str, Any], openai_api_key: str) -> Dict[str, Any]:
    """
    Use LangChain and OpenAI to enrich metadata with additional insights
    """
    try:
        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst helping to analyze tabular data structure. 
            Important requirements for your response:
            1. Always use null (not NaN, NA, or undefined) for missing values
            2. Ensure all numeric values are valid JSON numbers (no infinity or NaN)
            3. All strings must be properly escaped
            4. Arrays must contain only valid JSON values
            
            Analyze the provided metadata and identify:
            1. Potential primary keys (columns with unique, non-null values)
            2. Possible foreign key relationships between columns
            3. Column descriptions and purposes
            4. Data quality observations (including null patterns)
            5. Suggested data type improvements
            6. If multiple sheets exist, analyze relationships between sheets
            
            Return your analysis as a JSON object with these exact top-level keys:
            {
                "primary_key_candidates": [],
                "relationships": [],
                "column_descriptions": {},
                "data_quality": {},
                "type_suggestions": {},
                "sheet_relationships": {}  // Only for Excel files with multiple sheets
            }"""),
            ("user", "Analyze this metadata and provide insights following the format above: {metadata}")
        ])
        
        # Get LLM analysis
        chain = prompt | llm
        response = chain.invoke({"metadata": json.dumps(data["metadata"])})
        
        # Parse the LLM response and add it to the metadata
        llm_insights = json.loads(response.content)
        data["metadata"]["llm_insights"] = llm_insights
        
        return data
        
    except Exception as e:
        print(f"Warning: LLM enrichment failed: {str(e)}")
        return data

def convert_to_json(file_path: str, output_path: str, openai_api_key: str) -> None:
    """
    Main function to convert tabular data to JSON with enriched metadata
    """
    try:
        # Read file and get basic metadata
        data = read_tabular_file(file_path)
        
        # Enrich metadata using LLM
        enriched_data = enrich_metadata_with_llm(data, openai_api_key)
        
        # Save to JSON file using custom encoder
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            
        print(f"Successfully converted {file_path} to {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

# Example usage
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    convert_to_json(
        file_path="metadata.csv",  # Can be .csv, .xlsx, .xls, .tsv, or .txt
        output_path="metadata.json",
        openai_api_key=openai_api_key
    )