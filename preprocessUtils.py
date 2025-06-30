import os
import json
import pandas as pd
import openai
from tqdm import tqdm
import random
from collections import defaultdict
import yaml
import datetime

"""
******************************************************************
Utility functions to process CSV and TSV
******************************************************************
"""

def get_distinct_sample_values(df, column, n = 10):
    """Get up to n distinct values from a column"""
    try:
        # Only process the column if it exists in the dataframe
        if column in df.columns:
            distinct_values = df[column].dropna().unique()
            if len(distinct_values) > n:
                return random.sample(list(distinct_values), n)
            return list(distinct_values)
        return []
    except Exception as e:
        print(f"Error getting samples for column {column}: {str(e)}")
        return []

def infer_title_from_filename(filename):
    """Infer a title from the filename by replacing underscores with spaces and capitalizing words"""
    # Remove extension and path
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Replace underscores with spaces and capitalize words
    title = ' '.join(word.capitalize() for word in name_without_ext.split('_'))
    
    return title

def create_prompt(file_name, title, headers, sample_data):
    """Create a structured prompt for the LLM"""
    prompt = f"""Analyze this dataset and provide a structured analysis.

FILE INFORMATION:
- File name: {file_name}
- Title: {title}

HEADERS:
{', '.join(headers)}

COLUMN SAMPLES (up to 10 distinct values per column):
"""
    
    # Add sample data
    for column in headers:
        prompt += f"\n{column}: {sample_data[column]}"
    
    prompt += """

TASK:
Based on the data provided, generate a structured analysis with the following components:

RESPONSE FORMAT:
{
    "description": "A comprehensive description of what this dataset contains and represents",
    "column_descriptions": {
        "column1": "description of column1",
        "column2": "description of column2",
        ...
    }
}

RULES:
- Provide clear, concise descriptions for each field
- For column descriptions, include all columns from the headers
- In the description, explain what the dataset represents and its potential use cases
- Try to identify the type of data in each column (e.g., categorical, numerical, date, etc.)
"""
    
    return prompt

def get_llm_analysis(prompt, client):
    """Get analysis from LLM"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst specializing in datasets. You provide structured, accurate analyses of data. Always respond in JSON format with the exact structure requested, nothing more, nothing less."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            #response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Pulizia della risposta per assicurarsi che sia JSON valido
        # Rimuove eventuali backtick e indicatori di codice JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    except Exception as e:
        print(f"Error getting LLM analysis: {str(e)}")
        return {
            "description": "Error in LLM analysis",
            "column_descriptions": {},
            "spatial_coverage": "Unknown",
            "reasoning": f"Error: {str(e)}"
        }

def process_csv_file(client, csv_path, output_dir, sample_rows=1000):
    """Process a CSV/TSV file and generate metadata"""
    try:
        # Get file name without extension
        file_name = os.path.basename(csv_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        file_name = os.path.splitext(file_name)[0]
        
        # Determine separator based on file extension
        sep = '\t' if file_ext == '.tsv' else ','
        
        # First read only the headers
        headers = pd.read_csv(csv_path, sep = sep, nrows=0).columns.tolist()
        
        # Read only a sample of rows for efficiency
        df_sample = pd.read_csv(csv_path, sep = sep, nrows = sample_rows)
        
        # Infer title from filename
        title = infer_title_from_filename(file_name)
        
        # Get sample values for each column
        sample_data = {}
        column_examples = {}
        for column in headers:
            samples = get_distinct_sample_values(df_sample, column)
            sample_data[column] = samples
            # Convert samples to strings for examples
            column_examples[column] = [str(s) for s in samples]
        
        # Create prompt for LLM
        prompt = create_prompt(file_name, title, headers, sample_data)
        
        # Get LLM response
        llm_response = get_llm_analysis(prompt, client)
        
        # Combine LLM response with column examples
        combined_data = {
            "title": title,
            "column_examples": column_examples,
            "description": llm_response.get("description", ""),
            "column_descriptions": llm_response.get("column_descriptions", {})
        }
        
        # Save the result
        output_file = os.path.join(output_dir, f"{file_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        
        return output_file
    
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

"""
******************************************************************
Utility functions to make the Schema
******************************************************************
"""

def find_latest_processor_run(dir):
    """Find the latest run directory from the processor script"""
    preprocess_dir = os.path.join(dir)
    run_dirs = [d for d in os.listdir(preprocess_dir) if d.startswith("run_") and os.path.isdir(os.path.join(preprocess_dir, d))]
    
    if not run_dirs:
        return None
    
    # Sort by timestamp (which is part of the directory name)
    latest_run = sorted(run_dirs)[-1]
    return os.path.join(preprocess_dir, latest_run)

def normalize_column_name(column):
    """Normalize column names by removing numerical suffixes"""
    # Check if the column name follows the pattern class.attribute.number
    parts = column.split('.')
    if len(parts) > 2 and parts[-1].isdigit():
        # Remove the numerical suffix
        return '.'.join(parts[:-1])
    return column

def collect_all_column_data(json_files):
    """Collect all column descriptions and examples from all JSON files"""
    all_columns = defaultdict(lambda: {"descriptions": [], "examples": set()})
    
    for json_file in tqdm(json_files, desc="Collecting column data"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            column_descriptions = data.get('column_descriptions', {})
            column_examples = data.get('column_examples', {})
            
            for column, description in column_descriptions.items():
                if description and description.strip():  # Only add non-empty descriptions
                    # Normalize the column name to remove suffixes like .1, .2, etc.
                    normalized_column = normalize_column_name(column)
                    all_columns[normalized_column]["descriptions"].append(description)
            
            for column, examples in column_examples.items():
                if examples:  # Only add non-empty examples
                    # Normalize the column name to remove suffixes like .1, .2, etc.
                    normalized_column = normalize_column_name(column)
                    all_columns[normalized_column]["examples"].update(examples)
        
        except Exception as e:
            print(f"Error reading {json_file}: {str(e)}")
    
    return all_columns

def batch_unify_descriptions(columns_with_multiple_descriptions, client):
    """Unify all column descriptions in a single batch request to LLM"""
    try:
        # Prepare the data structure for LLM
        columns_data = {}
        for column, descriptions in columns_with_multiple_descriptions.items():
            columns_data[column] = {
                "descriptions": descriptions
            }
        
        # Create the prompt with all columns
        prompt = """Unify descriptions for multiple columns. For each column, select or create a single unified description that best captures the essence of all descriptions provided for that column.

COLUMNS AND THEIR DESCRIPTIONS:
"""
        
        # Add each column's descriptions to the prompt
        for column, data in columns_data.items():
            prompt += f"\nCOLUMN: {column}\nDESCRIPTIONS:\n"
            for i, desc in enumerate(data["descriptions"], 1):
                prompt += f"{i}. {desc}\n"
        
        prompt += """
TASK:
For each column, provide a single unified description that best captures the essence of all provided descriptions.
The unified descriptions should be clear, comprehensive, and concise.

RESPONSE FORMAT:
{
    "unified_descriptions": {
        "column1": "unified description for column1",
        "column2": "unified description for column2",
        ...
    }
}
"""
        print("Prompt:\n")
        print(prompt)
        # Make a single API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data modeling expert specializing in creating unified descriptions for data columns. Always respond in JSON format with the exact structure requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        unified_descriptions = result.get("unified_descriptions", {})
        
        # Validate that we got descriptions for all columns
        missing_columns = set(columns_with_multiple_descriptions.keys()) - set(unified_descriptions.keys())
        if missing_columns:
            print(f"Warning: Missing unified descriptions for columns: {missing_columns}")
            # Add default descriptions for missing columns by joining their descriptions
            for column in missing_columns:
                unified_descriptions[column] = "; ".join(columns_with_multiple_descriptions[column])
        
        return unified_descriptions
    
    except Exception as e:
        print(f"Error in batch unifying descriptions: {str(e)}")
        # Fallback: Join descriptions for each column
        fallback_descriptions = {}
        for column, descriptions in columns_with_multiple_descriptions.items():
            fallback_descriptions[column] = "; ".join(descriptions)
        return fallback_descriptions

def unify_column_data(all_columns, client):
    """Unify descriptions for columns that have multiple descriptions"""
    unified_columns = {}
    
    # Create a dictionary to hold columns with multiple descriptions
    columns_with_multiple_descriptions = {}
    
    # First, process all columns and identify which ones have multiple descriptions
    for column, data in all_columns.items():
        descriptions = data["descriptions"]
        examples = list(data["examples"])
        
        # Limit examples to 10 for each column
        if len(examples) > 10:
            examples = examples[:10]
        
        if len(descriptions) == 1:
            # If only one description, use it directly
            unified_columns[column] = {
                "description": descriptions[0],
                "examples": examples
            }
        else:
            # If multiple descriptions, add to the collection for batch processing
            columns_with_multiple_descriptions[column] = descriptions
    
    # If we have columns with multiple descriptions, process them all at once
    if columns_with_multiple_descriptions:
        print(f"Found {len(columns_with_multiple_descriptions)} columns with multiple descriptions")
        print("Processing all columns with multiple descriptions in a single batch...")
        
        # Get unified descriptions for all columns in one call
        unified_descriptions = batch_unify_descriptions(columns_with_multiple_descriptions, client)
        
        # Update the unified_columns dictionary with the batch results
        for column, unified_description in unified_descriptions.items():
            unified_columns[column] = {
                "description": unified_description,
                "examples": list(all_columns[column]["examples"])[:10]
            }
    
    return unified_columns

def extract_classes_from_columns(columns):
    """Extract class names and their attributes from column names"""
    classes = defaultdict(list)
    
    for column in columns:
        parts = column.split('.')
        if len(parts) > 1:
            # If format is class.attribute, add to that class
            class_name = parts[0]
            attribute = '.'.join(parts[1:])
            classes[class_name].append(attribute)
        else:
            # If no class prefix, add to a general Document class
            classes["Document"].append(column)
    
    return classes

def generate_linkml_schema(unified_columns, client):
    """Generate a LinkML schema from unified column descriptions and examples"""
    try:
        # Identify classes from column names
        classes = extract_classes_from_columns(unified_columns.keys())
        print(f"Identified classes: {', '.join(classes.keys())}")
        
        # Create prompt for LLM to generate the schema
        prompt = """Create a LinkML schema for a dataset with multiple classes.

IDENTIFIED CLASSES AND THEIR ATTRIBUTES:
"""
        for class_name, attributes in classes.items():
            prompt += f"\nCLASS: {class_name}\nATTRIBUTES: {', '.join(attributes)}\n"
        
        prompt += "\nDETAILED COLUMN DATA (description and examples):\n"
        for column, data in unified_columns.items():
            description = data["description"]
            examples = data["examples"]
            prompt += f"\n{column}:\n  Description: {description}\n  Examples: {', '.join(examples)}"
        
        prompt += """

TASK:
Create a comprehensive LinkML schema that includes all identified classes and their attributes.

RULES:
1. Follow the LinkML schema format as shown in the example below
2. Create a class for each identified class in the data
3. All classes should inherit from "NamedEntity"
4. For attributes that don't have a class prefix (like 'attribute' vs 'class.attribute'), 
   add them to a "Document" class that also inherits from "NamedEntity"
5. Use appropriate data types for each attribute based on the examples provided (string, integer, float, etc.)
6. Include clear descriptions for each attribute
7. Include examples for each attribute using the provided examples
8. Group related attributes together in the YAML structure
9. Use standard LinkML prefixes and imports

EXAMPLE FORMAT:
```yaml
id: https://w3id.org/schema
name: dataset-schema
title: Dataset Schema
description: >-
  A schema for a dataset.
prefixes:
  linkml: https://w3id.org/linkml/
  core: http://w3id.org/ontogpt/core/

default_range: string

imports:
  - linkml:types
  - core

classes:
  Document:
    is_a: NamedEntity
    description: A document representing a dataset
    attributes:
      attribute1:
        description: Description of attribute1
        range: string
        examples: example1, example2
  
  Gene:
    is_a: NamedEntity
    description: A gene entity
    attributes:
      type:
        description: The type of gene
        range: string
        examples: protein-coding, ncRNA
      symbol:
        description: The gene symbol
        range: string
        examples: BRCA1, TP53
```

RESPONSE FORMAT:
{
    "schema": "Your complete LinkML schema in YAML format here (without the ```yaml and ``` markers)"
}
"""
        
        # Print the prompt for debugging
        print("Schema generation prompt:")
        print("=" * 50)
        print(prompt[:500] + "... [truncated]")
        print("=" * 50)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data modeling expert specializing in LinkML schemas. You create well-structured, comprehensive schemas following LinkML conventions. Always respond in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        schema_yaml = result.get("schema", "")
        
        # Validate the schema to ensure it's compatible with the parser
        try:
            # yaml_data = yaml.safe_load(schema_yaml)
            yaml_data = yaml.load(schema_yaml, Loader=yaml.FullLoader)
            
            # Check if classes exist
            if "classes" not in yaml_data:
                print("Warning: Schema does not contain any classes")
                return schema_yaml
            
            # Ensure all classes have is_a: NamedEntity
            for class_name, class_data in yaml_data["classes"].items():
                if "is_a" not in class_data:
                    print(f"Warning: Adding 'is_a: NamedEntity' to {class_name} class")
                    yaml_data["classes"][class_name]["is_a"] = "NamedEntity"
                    
                    # Convert back to YAML
                    schema_yaml = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
            
        except Exception as e:
            print(f"Warning: Could not validate schema: {str(e)}")
        
        return schema_yaml
    
    except Exception as e:
        print(f"Error generating LinkML schema: {str(e)}")
        return ""

""" 
**************************************************************************************************
EUROSTAT
**************************************************************************************************
"""

def format_date(date_str):
    """Convert various date formats to yyyy-mm-ddThh:mm"""
    if not date_str or pd.isna(date_str):
        return None
    
    try:
        # Gestione speciale per date trimestrali (es. 2024-Q3)
        if isinstance(date_str, str) and 'Q' in date_str:
            # Estrai anno e trimestre
            parts = date_str.split('-Q')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                year = int(parts[0])
                quarter = int(parts[1])
                # Converti il trimestre in mese (Q1=01, Q2=04, Q3=07, Q4=10)
                month = (quarter - 1) * 3 + 1
                # Crea una data con il primo giorno del trimestre
                dt = datetime(year, month, 1)
                return dt.strftime('%Y-%m-%dT%H:%M')
        
        # Gestione speciale per date in formato YYYY-MM (es. 2005-03)
        if isinstance(date_str, str) and len(date_str) == 7 and date_str[4] == '-':
            try:
                # Verifica se è nel formato YYYY-MM
                year_str, month_str = date_str.split('-')
                if year_str.isdigit() and month_str.isdigit():
                    year = int(year_str)
                    month = int(month_str)
                    if 1 <= month <= 12:
                        dt = datetime(year, month, 1)
                        return dt.strftime('%Y-%m-%dT%H:%M')
            except ValueError:
                pass
        
        # Gestione per date in formato semestrale (es. 2023-S1, 2023-H1)
        if isinstance(date_str, str) and (('S1' in date_str) or ('S2' in date_str) or ('H1' in date_str) or ('H2' in date_str)):
            try:
                # Estrai anno e semestre
                if '-S' in date_str:
                    parts = date_str.split('-S')
                else:
                    parts = date_str.split('-H')
                
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    year = int(parts[0])
                    semester = int(parts[1])
                    # Converti il semestre in mese (S1/H1=01, S2/H2=07)
                    month = 1 if semester == 1 else 7
                    dt = datetime(year, month, 1)
                    return dt.strftime('%Y-%m-%dT%H:%M')
            except ValueError:
                pass
        
        # Gestione per date in formato YYYYMMDD (es. 20230101)
        if isinstance(date_str, str) and len(date_str) == 8 and date_str.isdigit():
            try:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    dt = datetime(year, month, day)
                    return dt.strftime('%Y-%m-%dT%H:%M')
            except ValueError:
                pass
        
        # Gestione per date in formato MM/YYYY o MM-YYYY
        if isinstance(date_str, str) and len(date_str) == 7 and (date_str[2] == '/' or date_str[2] == '-'):
            try:
                if date_str[2] == '/':
                    month_str, year_str = date_str.split('/')
                else:
                    month_str, year_str = date_str.split('-')
                
                if month_str.isdigit() and year_str.isdigit():
                    month = int(month_str)
                    year = int(year_str)
                    if 1 <= month <= 12:
                        dt = datetime(year, month, 1)
                        return dt.strftime('%Y-%m-%dT%H:%M')
            except ValueError:
                pass
        
        # Gestione per date in formato testuale (es. "Jan 2023", "January 2023")
        if isinstance(date_str, str) and any(month in date_str for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
            try:
                for fmt in ['%b %Y', '%B %Y', '%b, %Y', '%B, %Y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime('%Y-%m-%dT%H:%M')
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # Try different date formats
        for fmt in [
            '%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y', '%Y', 
            '%d.%m.%Y', '%m/%d/%Y', '%Y.%m.%d', '%d %b %Y', '%d %B %Y',
            '%b %d, %Y', '%B %d, %Y', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M',
            '%d/%m/%Y %H:%M', '%d/%m/%Y %H:%M:%S'
        ]:
            try:
                dt = datetime.strptime(str(date_str), fmt)
                return dt.strftime('%Y-%m-%dT%H:%M')
            except ValueError:
                continue
        
        # If none of the formats match, return None
        return None
    except Exception:
        return None
        
def process_csv_json_pair(client, csv_path, json_path, output_dir):
    """Process a CSV file and its associated JSON metadata"""
    try:
        # Get file name without extension
        file_name = os.path.basename(csv_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        file_name = os.path.splitext(file_name)[0]
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Read JSON metadata
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract relevant fields from metadata
        title = metadata.get('title', '')
        
        # Format dates from the original JSON only
        start_date = format_date(metadata.get('startDate', ''))
        end_date = format_date(metadata.get('endDate', ''))
        coverage_sector = metadata.get('coverageSector', '')
        
        # Get data themes
        data_themes = metadata.get('dataThemes', '')
        if not data_themes:
            data_themes = metadata.get('General', '')
        if not data_themes:
            data_themes = " - "
        
        # Get headers and sample values
        headers = list(df.columns)
        
        # Get sample values for each column
        sample_data = {}
        column_examples = {}
        for column in headers:
            samples = get_distinct_sample_values(df, column)
            sample_data[column] = samples
            # Convert samples to strings for examples
            column_examples[column] = [str(s) for s in samples]
        
        # Extract column descriptions from JSON, excluding those with " - " as description
        column_descriptions = {}
        for key, value in metadata.items():
            if key in headers and value != " - ":
                column_descriptions[key] = value
        
        # Create prompt for LLM
        prompt = create_prompt_eurostat(file_name, title, headers, sample_data, column_descriptions, 
                             start_date, end_date, coverage_sector)
        
        # Get LLM response
        llm_response = get_llm_analysis(prompt, client)
        
        # Combine LLM response with original metadata fields and column examples
        combined_data = {
            "title": title,
            "dataThemes": data_themes,
            "startDate": start_date,
            "endDate": end_date,
            "column_examples": column_examples,
            "description": llm_response.get("description", ""),
            "column_descriptions": llm_response.get("column_descriptions", {}),
            "spatial_coverage": llm_response.get("spatial_coverage", ""),
            "reasoning": llm_response.get("reasoning", "")
        }
        
        # Save the result
        output_file = os.path.join(output_dir, f"{file_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2)
        
        return output_file
    
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

def create_prompt_eurostat(file_name, title, headers, sample_data, column_descriptions, 
                 start_date, end_date, coverage_sector):
    """Create a structured prompt for the LLM"""
    prompt = f"""Analyze this Eurostat dataset and provide a structured analysis.

FILE INFORMATION:
- File name: {file_name}
- Title: {title}
"""

    # Add temporal information if available
    if start_date or end_date:
        prompt += "\nTEMPORAL INFORMATION:\n"
        if start_date:
            prompt += f"- Start date: {start_date}\n"
        if end_date:
            prompt += f"- End date: {end_date}\n"
    
    # Add coverage sector if available
    if coverage_sector:
        prompt += f"\nCOVERAGE SECTOR:\n{coverage_sector}\n"

    prompt += f"""
HEADERS:
{', '.join(headers)}

COLUMN SAMPLES (up to 10 distinct values per column):
"""
    
    # Add sample data
    for column in headers:
        prompt += f"\n{column}: {sample_data[column]}"
    
    # Add column descriptions if available
    if column_descriptions:
        prompt += "\n\nCOLUMN DESCRIPTIONS (from metadata):"
        for column, description in column_descriptions.items():
            prompt += f"\n{column}: {description}"
    
    prompt += """

TASK:
Based on the data provided, generate a structured analysis with the following components:

RESPONSE FORMAT:
{
    "description": "A comprehensive description of what this dataset contains and represents",
    "column_descriptions": {
        "column1": "description of column1",
        "column2": "description of column2",
        ...
    },
    "spatial_coverage": "The geographical areas covered by this dataset",
    "reasoning": "Explanation of your analysis and how you determined the descriptions and coverage"
}

RULES:
- Provide clear, concise descriptions for each field
- For column descriptions, include all columns from the headers
- Base your spatial coverage on country related columns if present, try to generalize an area
- Use the coverage sector information to enhance your understanding of the dataset
- In the reasoning section, explain your thought process and any patterns you observed
"""
    
    return prompt