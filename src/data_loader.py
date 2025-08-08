import os
import json
import pandas as pd
import random
import re
from datetime import datetime
from collections import Counter, defaultdict
import tiktoken
from tqdm import tqdm
from openai import OpenAI

from utils import add_timestamp, color_text, create_directory_if_not_exists

class DataLoader:
    """
    Handles loading and preprocessing of CSV/TSV files, with optional JSON metadata.
    It generates a standardized data lake representation (list of dictionaries)
    including basic stats and initial LLM-based descriptions.
    """

    def __init__(self,
        openai_client: OpenAI = None
    ):
        """
        Initializes DataLoader with optional OpenAI client.

        If OpenAI client is provided, it also loads the tiktoken tokenizer
        for generating LLM-based descriptions.

        Args:
            openai_client (OpenAI): An initialized OpenAI client instance.

        If no OpenAI client is provided, it prints a warning message and
        disables generation of LLM descriptions.
        """
        self.openai_client = openai_client
        self.tokenizer = None
        if openai_client: # Only load tokenizer if OpenAI client is provided
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(add_timestamp(color_text(f"Warning: Could not load tiktoken tokenizer: {e}", "yellow")))
        else:
            print(add_timestamp(color_text("Warning: OpenAI client not provided to DataLoader. LLM descriptions will be skipped.", "yellow")))

    def _truncate_text(self,
        text: str,
        max_tokens: int,
        model_name: str = 'text-embedding-ada-002'
    ) -> str:
        """
        Private helper method to truncate text to fit within the token limit for the embedding model.

        Args:
            text (str): The text to truncate.
            max_tokens (int): The maximum number of tokens allowed for the embedding model.
            model_name (str): The name of the embedding model to use.

        Returns:
            str: The truncated text, or the original text if it is already within the token limit.
        """
        if self.tokenizer:
            encoded_text = self.tokenizer.encode(text)
        else:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            encoded_text = encoding.encode(text)

        if len(encoded_text) > max_tokens:
            truncated_text = self.tokenizer.decode(encoded_text[:max_tokens]) if self.tokenizer else encoding.decode(encoded_text[:max_tokens])
            print(add_timestamp(color_text(f"Truncated text from {len(encoded_text)} to {max_tokens} tokens.", "yellow")))
            return truncated_text
        return text

    def _format_date(self,
        date_str: str
    ) -> str | None:
        """
        Private helper method to convert date strings to standardized format.

        This method takes a string date_str and attempts to convert it to
        a standardized format of 'YYYY-MM-DDTHH:MM' using various date formats.
        If the conversion is successful, it returns the converted date string.
        Otherwise, it returns None.

        This method is used internally by the DataLoader to convert date columns
        to a standardized format. The supported date formats are:
            - YYYY-MM-DDTHH:MM (ISO 8601 format)
            - YYYY-MM-DD (date only)
            - YYYY-MM (date only, with month as a number)
            - YYYY-QX (quarterly dates, where X is 1-4)
            - YYYY-SX or YYYY-HX (semester dates, where X is 1 or 2)
            - YYYYMMDD (date only, with month and day as numbers)
            - MM/YYYY or MM-YYYY (date only, with month as a number)
            - Jan 2023, January 2023, etc. (textual dates)

        If the input date string does not match any of the supported formats,
        it returns None.
        """
        if not date_str or pd.isna(date_str):
            return None

        date_str = str(date_str).strip()

        try:
            # Handle quarterly dates (e.g., 2024-Q3)
            if 'Q' in date_str and len(date_str.split('-Q')) == 2 and date_str.split('-Q')[0].isdigit() and date_str.split('-Q')[1].isdigit():
                year, quarter = map(int, date_str.split('-Q'))
                month = (quarter - 1) * 3 + 1
                dt = datetime(year, month, 1)
                return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle YYYY-MM (e.g., 2005-03)
            if len(date_str) == 7 and date_str[4] == '-':
                year, month = map(int, date_str.split('-'))
                if 1 <= month <= 12:
                    dt = datetime(year, month, 1)
                    return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle semester dates (e.g., 2023-S1, 2023-H1)
            if any(s in date_str for s in ['-S1', '-S2', '-H1', '-H2']):
                parts = re.split(r'-[SH]', date_str)
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    year, semester = int(parts[0]), int(parts[1])
                    month = 1 if semester == 1 else 7
                    dt = datetime(year, month, 1)
                    return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle YYYYMMDD (e.g., 20230101)
            if len(date_str) == 8 and date_str.isdigit():
                year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    dt = datetime(year, month, day)
                    return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle MM/YYYY or MM-YYYY
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

            # Handle textual dates (e.g., "Jan 2023", "January 2023")
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
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%dT%H:%M')
                except ValueError:
                    continue

            return None # If none of the formats match
        except Exception as e:
            print(add_timestamp(color_text(f"Error formatting date '{date_str}': {e}", "red")))
            return None

    def _get_distinct_sample_values(self,
        df: pd.DataFrame,
        column: str,
        n: int = 100
    ) -> list[str]:
        """
        Retrieves up to 'n' distinct non-null values from the specified column in a DataFrame.

        This method extracts unique values from the given column, excluding any null values.
        If the number of distinct values exceeds 'n', a random sample of 'n' values is returned.
        Otherwise, all distinct values are returned.

        Args:
            df (pd.DataFrame): The DataFrame from which to extract distinct values.
            column (str): The column name to process within the DataFrame.
            n (int, optional): The maximum number of distinct values to return. Defaults to 100.

        Returns:
            list[str]: A list of up to 'n' distinct non-null values from the specified column.
        """
        distinct_values = df[column].dropna().unique()
        if len(distinct_values) > n:
            return random.sample(list(distinct_values), n)
        return list(distinct_values)

    def _create_llm_description_prompt(self,
        file_info: dict
    ) -> str:
        """
        Creates a structured prompt for the LLM to analyze and describe a dataset.

        This method constructs a textual prompt that provides various details about a dataset,
        including file information, temporal coverage, headers, column samples, and metadata 
        descriptions if available. The prompt is used to instruct the LLM to generate a 
        comprehensive analysis of the dataset.

        Args:
            file_info (dict): A dictionary containing information about the dataset file, 
                            including file name, title, headers, column samples, metadata 
                            descriptions, temporal start and end dates, and coverage sector.

        Returns:
            str: A structured prompt string to be used by the LLM for dataset analysis.
        """

        file_name = file_info['file_name']
        title = file_info.get('title', '')
        headers = file_info['original_headers']
        sample_data = {col['column_name']: col['values_sample'] for col in file_info['columns']}
        json_column_descriptions = file_info.get('metadata_description', False)
        start_date = file_info.get('temporal_start_date', None)
        end_date = file_info.get('temporal_end_date', None)
        coverage_sector = file_info.get('coverage_sector', '')

        prompt = f"""Analyze this dataset and provide a structured analysis.

FILE INFORMATION:
- File name: {file_name}
- Title: {title}
"""

        if start_date or end_date:
            prompt += "\nTEMPORAL INFORMATION:\n"
            if start_date:
                prompt += f"- Start date: {start_date}\n"
            if end_date:
                prompt += f"- End date: {end_date}\n"

        if coverage_sector:
            prompt += f"\nCOVERAGE SECTOR:\n{coverage_sector}\n"

        prompt += f"""
HEADERS:
{', '.join(headers)}

COLUMN SAMPLES (up to 10 distinct values per column):
"""
        for column in headers:
            prompt += f"\n{column}: {sample_data.get(column, [])}"
            

        if json_column_descriptions:
            prompt += "\n\nCOLUMN DESCRIPTIONS (from metadata, if available):"
            for column in file_info["columns"]:
                description = column["metadata_description"]
                header = column["header"]
                prompt += f"\n{header}: {description}"

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
- Provide clear, concise descriptions for each field.
- For column descriptions, include all columns from the headers, even if no metadata description was provided.
- Base your spatial coverage on country related columns if present, try to generalize an area.
- Use the coverage sector information to enhance your understanding of the dataset.
- In the reasoning section, explain your thought process and any patterns you observed.
"""
        return prompt

    def _get_llm_description(self,
        prompt: str
    ) -> dict:
        """
        Obtains a structured analysis of a dataset using a Large Language Model (LLM).

        Args:
            prompt (str): The structured prompt to use for the LLM analysis.

        Returns:
            dict: A dictionary with the following keys: description, column_descriptions, spatial_coverage, reasoning.
        """
        if not self.openai_client:
            print(add_timestamp(color_text("OpenAI client not initialized. Skipping LLM description analysis.", "red")))
            return {
                "description": "LLM skipped: OpenAI client not initialized.",
                "column_descriptions": {},
                "spatial_coverage": "LLM skipped: OpenAI client not initialized.",
                "reasoning": "LLM skipped: OpenAI client not initialized."
            }
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o", # Can be made configurable
                messages=[
                    {"role": "system", "content": "You are a data analyst specializing in statistical datasets. You provide structured, accurate analyses of data. Always respond in the exact JSON format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            res = json.loads(response.choices[0].message.content)
            return res
        except Exception as e:
            print(add_timestamp(color_text(f"Error getting LLM analysis: {e}", "red")))
            return {
                "description": f"Error in LLM analysis: {e}",
                "column_descriptions": {},
                "spatial_coverage": "Unknown",
                "reasoning": f"Error: {e}"
            }
    
    def _get_unified_description(self,
        column: str,
        descriptions: list[str]
    ) -> str:
        """
        Unify multiple descriptions for a single column into a single comprehensive description.

        Args:
            column (str): The column name for the descriptions to be unified.
            descriptions (list[str]): A list of descriptions for the column.

        Returns:
            str: A single unified description for the column. If an error occurs, returns a semicolon joined list of the input descriptions.
        """
        try:
            prompt = f"""Unify the following descriptions for the column "{column}" into a single comprehensive description:

DESCRIPTIONS:
"""
            for i, desc in enumerate(descriptions, 1):
                prompt += f"{i}. {desc}\n"

            prompt += """
TASK:
Create a single unified description that captures the essence of all these descriptions.
The unified description should be clear, comprehensive, and concise.

RESPONSE FORMAT:
{
    "unified_description": "Your unified description here"
}
"""
            response = self.openai_client.chat.completions.create(
                model="gpt-4o", # Can be made configurable
                messages=[
                    {"role": "system", "content": "You are a data modeling expert specializing in creating unified descriptions for data columns. Always respond in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("unified_description", "; ".join(descriptions)) # Fallback
        except Exception as e:
            print(add_timestamp(color_text(f"Error unifying descriptions for {column}: {e}", "red")))
            return "; ".join(descriptions) # Fallback to joining descriptions
    
    def _unify_column_data(self,
        data_lake_list: list[dict]
    ) -> list[dict]:
        """
        Unifies column descriptions in the provided data lake list using an LLM.

        This function consolidates multiple descriptions for each column across the data lake
        into a single unified description using a Large Language Model (LLM). It updates
        each column's description with the unified result and returns the modified data lake list.

        Args:
            data_lake_list (list[dict]): A list of dictionaries representing files in the data lake,
                                        each containing column data with descriptions.

        Returns:
            list[dict]: The updated data lake list with unified column descriptions.
        """
        print(add_timestamp(color_text("Unifying column data...", "yellow")))
        all_columns = defaultdict(list)

        # Collect all column data
        for file_dict in data_lake_list:
            for column_dict in file_dict['columns']:
                column_header = column_dict['header']
                if column_dict.get('llm_description'):
                    all_columns[column_header].append(column_dict['llm_description'])

        unified_columns = {}
        for column_header, descriptions in tqdm(all_columns.items(), desc="Unifying descriptions with LLM"):
            if len(set(descriptions)) > 1:
                # Use LLM to unify descriptions if there are multiple unique ones
                unified_desc = self._get_unified_description(column_header, list(set(descriptions)))
            elif descriptions:
                unified_desc = descriptions[0]
            else:
                unified_desc = "N/A"

            unified_columns[column_header] = unified_desc
        
        # Update the original data_lake_list with the unified data
        for file_dict in data_lake_list:
            for column_dict in file_dict['columns']:
                column_header = column_dict['header']
                unified_data = unified_columns.get(column_header)
                if unified_data:
                    column_dict['llm_description'] = unified_data

        print(add_timestamp(color_text("Column data unification complete.", "green")))
        return data_lake_list
    
    def load_and_describe_datalake(self,
        data_directory: str,
        metadata_directory: str | None = None,
        llm: bool = True,
        sample_size: int = 100,
        max_prompt_tokens: int = 8000,
        output_directory: str | None = None
    ) -> list[dict]:
        """
        Loads data from CSV/TSV files in the provided data directory and creates a
        data lake list with descriptions generated using a Large Language Model (LLM).
        
        Args:
            data_directory (str): Path to the directory containing CSV/TSV files.
            metadata_directory (str | None): Path to the directory containing JSON metadata
                                             files matching the data files (optional).
            llm (bool): Whether to use an LLM to generate descriptions for files and columns.
            sample_size (int): Number of distinct values to sample from each column for statistics.
            max_prompt_tokens (int): Maximum number of tokens to use in the LLM prompt.
            output_directory (str | None): Directory to save the generated data lake JSON file to.
        
        Returns:
            list[dict]: A list of dictionaries, each representing a data file with its columns,
                        containing statistical information and descriptions generated by the LLM.
        """
        print(add_timestamp(color_text(f"Starting data lake loading and description from {data_directory}", "magenta")))
        data_lake_list = []

        csv_files = [f for f in os.listdir(data_directory) if f.endswith(('.csv', '.tsv'))]
        if not csv_files:
            print(add_timestamp(color_text(f"No CSV/TSV files found in {data_directory}.", "yellow")))
            return []

        for file_name in tqdm(csv_files, desc="Processing data files"):
            file_path = os.path.join(data_directory, file_name)
            sep = '\t' if file_name.endswith(".tsv") else ','

            try:
                df = pd.read_csv(file_path, sep=sep, low_memory=False, dtype=str)
                if df.empty:
                    print(add_timestamp(color_text(f"Skipping empty file: {file_name}", "yellow")))
                    continue

                file_info = {
                    "file_name": file_name,
                    "file_path": file_path, # Store full path for later access if needed
                    "title": "",
                    "llm_description": "",
                    "spatial_coverage_llm": None,
                    "temporal_start_date": None,
                    "temporal_end_date": None,
                    "data_themes": None,
                    "coverage_sector": None,
                    "original_headers": list(df.columns),
                    "columns": [],
                    "metadata_description": False
                }
                
                metadata_column_descriptions = {}

                # --- Integrate Metadata (if provided) ---
                json_metadata = {}
                if metadata_directory:
                    # Try common naming conventions for metadata files
                    base_name_no_ext = os.path.splitext(file_name)[0]
                    metadata_candidates = [
                        f"{base_name_no_ext}.json",
                        f"{base_name_no_ext.replace('_data', '_metadata')}.json", # e.g., customers_data.csv -> customers_metadata.json
                    ]
                    found_metadata_path = None
                    for candidate in metadata_candidates:
                        potential_path = os.path.join(metadata_directory, candidate)
                        if os.path.exists(potential_path):
                            found_metadata_path = potential_path
                            break

                    if found_metadata_path:
                        try:
                            with open(found_metadata_path, 'r', encoding='utf-8') as f:
                                json_metadata = json.load(f)
                            print(add_timestamp(color_text(f"Found metadata for {file_name}: {os.path.basename(found_metadata_path)}", "blue")))

                            file_info["title"] = json_metadata.get('title', '')
                            file_info["temporal_start_date"] = self._format_date(json_metadata.get('startDate', ''))
                            file_info["temporal_end_date"] = self._format_date(json_metadata.get('endDate', ''))
                            file_info["data_themes"] = json_metadata.get('dataThemes', json_metadata.get('General', ''))
                            file_info["coverage_sector"] = json_metadata.get('coverageSector', '')
                            file_info["metadata_description"] = True

                            # Extract column descriptions from JSON metadata
                            for col_name in df.columns:
                                if col_name in json_metadata and json_metadata[col_name] and json_metadata[col_name] != " - ":
                                    metadata_column_descriptions[col_name] = json_metadata[col_name]

                        except Exception as e:
                            print(add_timestamp(color_text(f"Error loading/parsing metadata for {file_name} from {found_metadata_path}: {e}", "red")))
                    else:
                        print(add_timestamp(color_text(f"No matching metadata JSON found for {file_name} in {metadata_directory}", "yellow")))

                # --- Process Columns for Statistics and Samples ---
                for col_name in df.columns:
                    values = df[col_name].dropna().astype(str).tolist()
                    num_values = len(values)

                    # Basic statistics
                    if num_values > 0:
                        lengths = [len(v) for v in values]
                        max_length = max(lengths)
                        min_length = min(lengths)
                        avg_length = sum(lengths) / num_values
                        counter = Counter(values)
                        most_common_values = [val for val, _ in counter.most_common(min(20, num_values))] # Limit to 20
                    else:
                        max_length, min_length, avg_length = 0, 0, 0
                        most_common_values = []

                    column_dict = {
                        "column_name": col_name,
                        "header": col_name, # Original header
                        "values_sample": self._get_distinct_sample_values(df, col_name, sample_size),
                        "value_stats": {
                            "num_values": num_values,
                            "max_length": max_length,
                            "min_length": min_length,
                            "avg_length": round(avg_length, 2),
                            "most_common_values": most_common_values
                        },
                        "metadata_description": metadata_column_descriptions.get(col_name) or None,
                        "llm_description": None, # To be filled by LLM
                        "semantic_annotation": None, # To be filled by SemanticAnnotator later (label)
                    }
                    file_info["columns"].append(column_dict)

                if llm:
                    # --- Generate LLM Descriptions for File and Columns ---
                    llm_prompt = self._create_llm_description_prompt(file_info)
                    llm_prompt_truncated = self._truncate_text(llm_prompt, max_prompt_tokens)
                    llm_response = self._get_llm_description(llm_prompt_truncated)

                    file_info["llm_description"] = llm_response.get("description", "")
                    file_info["llm_spatial_coverage"] = llm_response.get("spatial_coverage", "")

                    for col_dict in file_info["columns"]:
                        col_name = col_dict["column_name"]
                        col_dict["llm_description"] = llm_response["column_descriptions"].get(col_name, "")
                
                data_lake_list.append(file_info)
                
            except Exception as e:
                print(add_timestamp(color_text(f"Error processing file {file_name}: {e}", "red")))
                continue
            
        if llm:
            data_lake_list = self._unify_column_data(data_lake_list)
            
        # --- Save Data Lake JSON ---
        if output_directory:
            try:
                create_directory_if_not_exists(output_directory)
                output_file = os.path.join(output_directory, "data_lake.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_lake_list, f, indent=2)
                print(add_timestamp(color_text(f"Data lake JSON saved to: {output_file}", "green")))
            except Exception as e:
                print(add_timestamp(color_text(f"Error saving data lake JSON to {output_file}: {e}", "red")))

        print(add_timestamp(color_text(f"Finished loading and describing {len(data_lake_list)} data files.", "green")))
        return data_lake_list
