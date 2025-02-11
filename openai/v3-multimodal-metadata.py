import base64
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import mimetypes
import fitz  # PyMuPDF for PDF
from PIL import Image
import pandas as pd

def encode_file_to_base64(file_path: str) -> tuple[str, str]:
    """
    Encode file to base64 and determine its MIME type
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    with open(file_path, 'rb') as file:
        base64_data = base64.b64encode(file.read()).decode('utf-8')
    return base64_data, mime_type

def convert_to_image(file_path: str) -> Optional[str]:
    """
    Convert various file types to image for vision model processing
    Returns base64 encoded image
    """
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext in ['.pdf']:
            # Convert first page of PDF to image
            doc = fitz.open(file_path)
            page = doc[0]
            pix = page.get_pixmap()
            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            # For spreadsheets, read and convert to image using DataFrame styling
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Limit to first 20 rows for visualization
            df_display = df.head(20)
            
            # Create styled DataFrame image
            styled = df_display.style.background_gradient(cmap='Blues')
            
            # Convert to image
            img_data = styled.to_image()
        
        else:
            # Try to open as image directly
            img_data = Image.open(file_path)
        
        # Convert to base64
        import io
        img_byte_arr = io.BytesIO()
        img_data.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')
        
    except Exception as e:
        print(f"Warning: Could not convert file to image: {str(e)}")
        return None

def analyze_file_with_gpt4v(file_path: str, openai_api_key: str) -> Dict[str, Any]:
    """
    Analyze file using GPT-4 Vision and convert to structured JSON
    """
    # Initialize GPT-4 Vision
    llm = ChatOpenAI(
        model="gpt-4-vision-preview",
        max_tokens=4096,
        temperature=0,
        api_key=openai_api_key
    )
    
    # Convert file to image
    base64_image = convert_to_image(file_path)
    
    # Prepare the message with the image
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data extraction specialist that converts documents to structured JSON.
        
        Requirements for your JSON output:
        1. Always use null for missing values (not NaN, NA, undefined)
        2. Ensure all numeric values are valid JSON numbers
        3. All strings must be properly escaped
        4. Arrays must contain only valid JSON values
        
        Analyze the provided document and extract:
        1. Basic metadata about the document
        2. All visible data in a structured format
        3. Column names and data types for tabular data
        4. Any visible relationships or hierarchies
        5. Any relevant headers, titles, or metadata
        
        Return your analysis as a JSON object with these exact top-level keys:
        {
            "metadata": {
                "document_type": "",
                "structure_type": "",  // e.g., "table", "form", "text", etc.
                "detected_fields": []
            },
            "content": {
                // Extracted content here
            },
            "schema": {
                // Data structure information
            },
            "relationships": []
        }"""),
        ("user", {
            "type": "image",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        } if base64_image else "Could not convert file to image. Please provide text analysis.")
    ])
    
    # Get analysis
    chain = prompt | llm
    response = chain.invoke({})
    
    # Parse the response
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # If the response isn't valid JSON, try to extract JSON from the text
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise Exception("Could not parse JSON from response")
    
    return result

def convert_file_to_json(file_path: str, output_path: str, openai_api_key: str) -> None:
    """
    Main function to convert any supported file to JSON with metadata
    """
    try:
        # Analyze file with GPT-4 Vision
        result = analyze_file_with_gpt4v(file_path, openai_api_key)
        
        # Add file metadata
        result["metadata"]["filename"] = os.path.basename(file_path)
        result["metadata"]["file_size"] = os.path.getsize(file_path)
        result["metadata"]["file_type"] = mimetypes.guess_type(file_path)[0]
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully converted {file_path} to {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

# Example usage
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    convert_file_to_json(
        file_path="document.pdf",  # Can be PDF, Excel, CSV, images, etc.
        output_path="output_with_metadata.json",
        openai_api_key=openai_api_key
    )