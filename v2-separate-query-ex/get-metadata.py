import os
import pandas as pd
import json

def read_and_merge_files(folder_path, output_file):
    metadata = {}
    
    # Ensure the metadata folder exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith((".csv", ".xls", ".xlsx")):
            try:
                # Read the file using pandas
                if file_name.lower().endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Store the file's content as a list of records under its filename key
                metadata[file_name] = df.to_dict(orient="records")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    
    # Write the combined metadata to a single JSON file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(metadata, out_f, indent=4)

# Example usage
folder_path = "/Users/rudrapratapmohanty/Desktop/projects/sql-gen/input_metadata"
output_file = "metadata/metadata.json"
read_and_merge_files(folder_path, output_file)
