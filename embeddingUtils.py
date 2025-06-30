from utils import count_tokens, truncate_text
from collections import Counter
import json
import os
import pandas as pd

def load_datasets(directory_path, method, semantic_annotations_path = None):
    if method == "semantic_annotation" and semantic_annotations_path is not None:
        # Load semantic annotations from JSON file
        with open(semantic_annotations_path, 'r') as f:
            semantic_annotations = json.load(f)
    
        # Create mapping of filename to column annotations
        annotations_map = {
            item['file_name']: item['column_annotations'] 
            for item in semantic_annotations
        }
    
    datasets = []
    for file_name in os.listdir(directory_path):
        # Filter for CSV/TSV files
        if not (file_name.endswith(".csv") or file_name.endswith(".tsv")):
            continue
        
        if method == "semantic_annotation" and semantic_annotations_path is not None:   
            # Skip if no annotations found for this file
            if file_name not in annotations_map:
                print(f"No semantic annotations found for file: {file_name}")
                continue
        
        file_path = os.path.join(directory_path, file_name)
        sep = '\t' if file_name.endswith(".tsv") else ','

        try:
            # Read the file into dataframe
            df = pd.read_csv(file_path, 
                          sep = sep,
                          encoding = 'utf-8',
                          on_bad_lines = 'skip',
                          dtype = str,  # Force all columns as strings for consistency
                          header = 0)  # Use first row as header unless pkt_flag is True
            original_headers = list(df.columns)
        except Exception as e:
            print(f"Error parsing CSV/TSV file {file_name}: {e}")
            continue

        if method == "semantic_annotation" and semantic_annotations_path is not None:
            # Get semantic annotations for this file
            file_annotations = annotations_map[file_name]
        
        # Create temporary column names (Column_1, Column_2, etc.) for JSON mapping
        temp_column_names = [f"Column_{i+1}" for i in range(len(df.columns))]
        
        # Map column names to their annotations
        column_annotations = {}
        for i in range(len(df.columns)):
            col_key = f"Column_{i+1}"
            
            if method == "semantic_annotation" and semantic_annotations_path is not None:
                if col_key in file_annotations:
                    annotation = file_annotations[col_key]
                    # If annotation contains "context", set to NA
                    if "context" in annotation.lower():
                        column_annotations[col_key] = "NA"
                    else:
                        column_annotations[col_key] = annotation
                else:
                    column_annotations[col_key] = ""
            else:
                column_annotations[col_key] = "NA"
                
        
        final_column_names = list(df.columns)
        
        # Handle duplicate column names by adding .1, .2, .3, etc.
        final_column_names_unique = []
        name_counts = {}
        
        for name in final_column_names:
            if name in name_counts:
                name_counts[name] += 1
                final_column_names_unique.append(f"{name}.{name_counts[name]}")
            else:
                name_counts[name] = 0
                final_column_names_unique.append(name)
        
        # Keep original column ordering for processing
        original_df = df.copy()
        original_df.columns = temp_column_names
        
        # Drop rows with all NaN values
        original_df.dropna(how='all', inplace=True)
        
        # Drop duplicate columns
        original_df = original_df.loc[:, ~original_df.columns.duplicated()]

        # Create columns data with annotations
        columns_data = [
            {
                'column_name': original_headers[i] if i < len(original_headers) else f"Column_{i+1}",
                # 'column_name': col_name,
                #'original_column_name': original_headers[i] if i < len(original_headers) else f"Column_{i+1}",
                'annotation': column_annotations.get(col_name, ""),
                'header': original_headers[i] if i < len(original_headers) else "",  # Use original headers
                'values': original_df[col_name].dropna().astype(str).tolist()
            }
            for i, col_name in enumerate(temp_column_names)
            if i < len(final_column_names_unique)
        ]

        # Append file data
        datasets.append({
            'file_name': file_name,
            'columns': columns_data
        })

    return datasets

def get_embedding(text, client):
    """Obtains embeddings from OpenAI using text-embedding-3-small model."""
    model="text-embedding-3-small"
    try:   
        response = client.embeddings.create(
            input = [text],
            model = model
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error retrieving the embedding for the text: {e}")
        return None
    
def prepare_text_with_semantic_annotation(column_dict, method, SAFE_MAX_TOKENS=8000, yaml_descriptions = None):
    try:
        col_name = column_dict['column_name']
        header = column_dict.get('header', "")
        values = column_dict['values']
        semantic_annotation = column_dict['annotation']

        num_values = len(values)

        if num_values > 0:
            lengths = [len(v) for v in values]
            max_length = max(lengths)
            min_length = min(lengths)
            avg_length = sum(lengths) / num_values
        else:
            max_length = 0
            min_length = 0
            avg_length = 0

        # Extract the most frequent values
        counter = Counter(values)
        most_common_values = [val for val, count in counter.most_common(10)]
        
        # Choose a subset of random values for examples
        random_values = []
        if len(values) > 10:
            import random
            random_values = random.sample(values, 10)
        else:
            random_values = values[:10]

        if method == "semantic_annotation" and yaml_descriptions is not None:
            # Semantic annotation is available - create rich prompt with class/attribute info
            if "." in semantic_annotation:
                class_name, attribute_name = semantic_annotation.split('.')[:2]
            else:
                class_name = semantic_annotation
                attribute_name = ""
                
            class_description = yaml_descriptions.get(class_name, "No description available")
            attribute_description = yaml_descriptions.get(semantic_annotation, "No description available")
            attribute_examples = ", ".join(most_common_values[:4])

            text_parts = [
                f"The attribute with header: '{header}' and semantic annotation: '{semantic_annotation}', belongs to instances of type '{class_name}' and is described as: '{attribute_description}'.",
                f"The class '{class_name}' can be described as: '{class_description}'.",
                f"Examples of values for this attribute include: '{attribute_examples}'.",
                "",
                f"The dataset column for '{semantic_annotation}' contains {num_values} entries.",
                "",
                "Key statistics for the column:",
                f"- Maximum value length: {max_length} characters.",
                f"- Minimum value length: {min_length} characters.",
                f"- Average value length: {avg_length:.1f} characters.",
                "",
                "Top 10 most frequent values in the column:",
                ", ".join(most_common_values)
            ]
        else:
            # No semantic annotation - focus on raw data characteristics
            # Detect the likely data type
            is_numeric = all(v.replace('.', '', 1).replace('-', '', 1).isdigit() for v in values[:100] if v)
            has_letters = any(any(c.isalpha() for c in v) for v in values[:100] if v)
            unique_ratio = len(counter) / max(num_values, 1)
            
            data_type = "unknown"
            if is_numeric:
                data_type = "numeric"
            elif unique_ratio > 0.9:
                data_type = "identifier/unique key"
            elif has_letters:
                data_type = "text"
                
            if header == "":
                text_parts = [
                    #f"Column with header: '{header}'",
                    f"This column contains {num_values} values with {len(counter)} unique values ({unique_ratio:.1%} uniqueness).",
                    #f"The data appears to be {data_type} in nature.",
                    "",
                    "Value characteristics:",
                    f"- Maximum length: {max_length} characters",
                    f"- Minimum length: {min_length} characters",
                    f"- Average length: {avg_length:.1f} characters",
                    "",
                ]
            else:
                text_parts = [
                    f"Column with header: '{header}'",
                    f"This column contains {num_values} values with {len(counter)} unique values ({unique_ratio:.1%} uniqueness).",
                    #f"The data appears to be {data_type} in nature.",
                    "",
                    "Value characteristics:",
                    f"- Maximum length: {max_length} characters",
                    f"- Minimum length: {min_length} characters",
                    f"- Average length: {avg_length:.1f} characters",
                    "",
                ]
            # if numerical_stats:
            #     text_parts.extend(["", "Numerical statistics:"] + numerical_stats)
                
            text_parts.extend([
                "",
                "Most frequent values:",
                ", ".join(most_common_values[:10]),
                "",
                "Random sample values:",
                ", ".join(random_values)
            ])

        text = '\n'.join(text_parts)
        
        # Print the prompt for debugging
        print(f"\n{'=' * 80}\nDEBUG - Generated prompt for {header or col_name}:\n{'-' * 80}\n{text}\n{'=' * 80}\n")
        
        # Check token count and truncate if needed
        num_tokens_used = count_tokens(text)
        if num_tokens_used > SAFE_MAX_TOKENS:
            text = truncate_text(text, SAFE_MAX_TOKENS)

        return text
    except Exception as e:
        print(f"Error preparing text for column {column_dict.get('column_name','?')}: {e}")
        return None