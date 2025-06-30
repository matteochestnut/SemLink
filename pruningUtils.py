import os
import openai
import json
import pandas as pd

def process_dataset(file_path):
    """
    Process a dataset to extract relevant information for classification
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        dict: Dictionary containing file metadata and sample data
    """
    try:
        # Get file name without path
        file_name = os.path.basename(file_path)
        
        # Read first few lines to check for empty line and header
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(3)]
        
        sep = '\t' if file_path.endswith('.tsv') else ','
        
        # Check if second line is empty
        has_header = len(first_lines) > 2 and not first_lines[1].strip()
        header = first_lines[2].strip().split(sep) if has_header else None
        
        # Read file
        skiprows = 3 if has_header else 1
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, skiprows=skiprows)
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t', skiprows=skiprows)
        else:
            raise ValueError(f"Unsupported file format for {file_path}")
        
        # Create masked column names
        masked_columns = [f"Column_{i+1}" for i in range(len(df.columns))]
        df.columns = masked_columns
        
        # Get 20 random rows for each column
        sample_data = {}
        for column in masked_columns:
            # Remove NaN values and get sample
            clean_column = df[column].dropna()
            if len(clean_column) > 20:
                sample_data[column] = clean_column.sample(n=20).tolist()
            else:
                sample_data[column] = clean_column.tolist()
        
        return {
            "file_name": file_name,
            "headers": header if has_header else None,
            "masked_columns": masked_columns,
            "sample_data": sample_data
        }
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def classify_columns(dataset_info, schema_descriptions, client):
    """
    Create prompt and get classification from ChatGPT to identify the most relevant
    schema classes for the entire document
    
    Args:
        dataset_info (dict): Dataset information from process_dataset
        schema_descriptions (dict): Schema class descriptions
        
    Returns:
        dict: Classification results
    """
    prompt = f"""Analyze this dataset as a whole document and identify which schema classes (maximum 3) best characterize its content and purpose.

    FILE INFORMATION:
    - File name: {dataset_info['file_name']}"""


    if dataset_info['headers']:
        prompt += f"\n- Headers: {', '.join(dataset_info['headers'])}"
    
    prompt += "\n\nCOLUMN SAMPLES (first 5 samples per column):"
    
    # Add sample data to prompt
    for column in dataset_info['masked_columns']:
        prompt += f"\n{column}: {dataset_info['sample_data'][column][:5]}"
    
    prompt += "\n\nSCHEMA CLASSES:"
    for class_name, description in schema_descriptions.items():
        prompt += f"\n{class_name}: {description}"
    
    prompt += """

    TASK:
    Identify the most relevant schema classes (maximum 3) that best characterize this document as a whole.

    RESPONSE FORMAT:
    {
        "file_name": "example.tsv",
        "relevant_classes": ["Class1", "Class2"],  // Maximum 3 classes, ordered by relevance
        "reasoning": [
            "1. Why this document primarily deals with Class1...",
            "2. Why Class2 is also relevant for this document...",
            "3. Why other classes are less relevant or not applicable..."
        ]
    }

    RULES:
    - Consider the document as a whole, not individual columns
    - Select maximum 3 classes that best describe the document's domain and purpose
    - Base your decision on:
    * The overall content and purpose of the dataset
    * The types of entities being described or linked
    * The domain context suggested by the file name and data
    - Provide clear explanations for why these classes are most relevant
    - Explain why other classes were not selected
    - Only select classes that are strongly represented in the document"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data scientist specialized in biomedical data classification. You always respond in the exact JSON format requested, with no additional text before or after."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Error getting classification: {str(e)}")
        return None