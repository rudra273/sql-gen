import psycopg2
import pandas as pd
import os
from datetime import datetime

def connect_to_db():
    """Establish database connection"""
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )

def get_all_tables(cursor):
    """Get list of all tables in the public schema"""
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE';
    """)
    return [table[0] for table in cursor.fetchall()]

def get_table_schema(table_name, cursor):
    """Get detailed schema information for a specific table"""
    # Get column information
    cursor.execute("""
        SELECT 
            c.column_name,
            c.data_type,
            c.is_nullable,
            c.column_default,
            c.character_maximum_length,
            c.numeric_precision,
            c.numeric_scale,
            CASE 
                WHEN pk.column_name IS NOT NULL THEN 'PRIMARY KEY'
                WHEN fk.column_name IS NOT NULL THEN 'FOREIGN KEY'
                ELSE ''
            END as key_type,
            fk.foreign_table_name,
            fk.foreign_column_name
        FROM information_schema.columns c
        LEFT JOIN (
            SELECT ku.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage ku
                ON tc.constraint_name = ku.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_name = %s
        ) pk ON c.column_name = pk.column_name
        LEFT JOIN (
            SELECT 
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu 
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = %s
        ) fk ON c.column_name = fk.column_name
        WHERE c.table_name = %s
        ORDER BY c.ordinal_position;
    """, (table_name, table_name, table_name))
    
    return cursor.fetchall()

def get_table_size(table_name, cursor):
    """Get the number of rows in a table"""
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    return cursor.fetchone()[0]

def format_schema_info(schema_info):
    """Format column information into a readable string"""
    (column_name, data_type, is_nullable, default, max_length, 
     num_precision, num_scale, key_type, foreign_table, foreign_column) = schema_info
    
    parts = []
    parts.append(f"Type: {data_type}")
    
    # Add length/precision information if applicable
    if max_length:
        parts.append(f"Length: {max_length}")
    if num_precision is not None and data_type.startswith('numeric'):
        parts.append(f"Precision: {num_precision}, Scale: {num_scale}")
    
    parts.append("Nullable" if is_nullable == 'YES' else "Not Nullable")
    
    if default:
        parts.append(f"Default: {default}")
    
    if key_type:
        parts.append(key_type)
        if key_type == 'FOREIGN KEY':
            parts.append(f"References {foreign_table}({foreign_column})")
    
    return " | ".join(parts)

def main():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Create directory for schema documentation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'schema_docs_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all tables
    tables = get_all_tables(cursor)
    
    # Create a summary DataFrame for all tables
    summary_data = []
    
    # Process each table
    for table in tables:
        print(f"Processing table: {table}")
        
        # Get schema information
        columns = get_table_schema(table, cursor)
        row_count = get_table_size(table, cursor)
        
        # Create DataFrame for table schema
        schema_data = []
        for col_info in columns:
            schema_data.append({
                'Column Name': col_info[0],
                'Details': format_schema_info(col_info)
            })
        
        df = pd.DataFrame(schema_data)
        
        # Save individual table schema
        csv_path = f'{output_dir}/{table}_schema.csv'
        df.to_csv(csv_path, index=False)
        
        # Add to summary
        summary_data.append({
            'Table Name': table,
            'Number of Columns': len(columns),
            'Number of Rows': row_count,
            'Primary Key': next((col[0] for col in columns if 'PRIMARY KEY' in col[7]), 'None'),
            'Foreign Keys': ', '.join([col[0] for col in columns if 'FOREIGN KEY' in col[7]])
        })
    
    # Create and save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/database_summary.csv', index=False)
    
    # Create relationships file
    relationships_data = []
    for table in tables:
        columns = get_table_schema(table, cursor)
        for col_info in columns:
            if col_info[7] == 'FOREIGN KEY':
                relationships_data.append({
                    'Source Table': table,
                    'Source Column': col_info[0],
                    'Referenced Table': col_info[8],
                    'Referenced Column': col_info[9]
                })
    
    if relationships_data:
        relationships_df = pd.DataFrame(relationships_data)
        relationships_df.to_csv(f'{output_dir}/table_relationships.csv', index=False)
    
    cursor.close()
    conn.close()
    
    print(f"\nSchema documentation has been generated in the '{output_dir}' directory")
    print("Files generated:")
    print("1. database_summary.csv - Overview of all tables")
    print("2. table_relationships.csv - All foreign key relationships")
    print("3. Individual schema files for each table")

if __name__ == "__main__":
    main()