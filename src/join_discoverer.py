import os
import json
import time
import numpy as np
import pandas as pd
import faiss
import tiktoken
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Union

from utils import add_timestamp, color_text, create_directory_if_not_exists, load_datalake_json


class JoinDiscoverer:
    """
    This class provides a streamlined workflow to:
    1. Generate embeddings for a data lake's columns using an LLM.
    2. Calculate various distance metrics between these embeddings.
    3. Use the distances to generate a Neo4j-compatible CSV for nodes and relationships.
    """

    def __init__(self, openai_client: OpenAI):
        """
        Initializes the DataJoinerPipeline.

        Args:
            openai_client (OpenAI): An initialized OpenAI client instance.
        """
        self.openai_client = openai_client
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(add_timestamp(color_text(f"Warning: Could not load tiktoken tokenizer: {e}", "yellow")))

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
            truncated_text = self.tokenizer.decode(encoded_text[:max_tokens])
            return truncated_text
        return text

    def _prepare_embedding_prompt(self, column_data: dict, mode: str, max_tokens: int) -> Union[str, None]:
        """
        The prompt is built conditionally based on the available data and the specified mode.
        This method now combines information from semantic annotations, LLM descriptions,
        and column value statistics for a richer context.

        Args:
            column_data (dict): A dictionary containing all information for a single column.
            mode (str): The mode to use for preparing the embedding prompt.
                        Options: 'semantic_mode', 'header_mode'.
            max_tokens (int): The maximum number of tokens allowed for the prompt.

        Returns:
            Union[str, None]: The prepared prompt string, or None if no valid data is found.
        """
        # Extract data from the column dictionary
        header = column_data.get('header', column_data.get('column_name', ''))
        semantic_annotation = column_data.get('semantic_annotation', 'NA')
        llm_description = column_data.get('llm_description', '')
        value_stats = column_data.get('value_stats', {})
        attribute_examples = ", ".join(column_data.get('values_sample', []))

        # Extract specific statistics
        num_values = value_stats.get('num_values', 0)
        max_length = value_stats.get('max_length', 0)
        min_length = value_stats.get('min_length', 0)
        avg_length = value_stats.get('avg_length', 0.0)
        most_common_values = value_stats.get('most_common_values', [])

        text_parts = []
        if mode == 'semantic_mode' and semantic_annotation != 'NA':
            try:
                # Assuming format is "class_name.attribute"
                class_name, _ = semantic_annotation.split('.', 1)
            except ValueError:
                class_name = 'NA'

            prompt_line = f"The attribute with header: '{header}' and semantic annotation: '{semantic_annotation}', belongs to instances of type '{class_name}'"
            if llm_description:
                prompt_line += f" and is described as: '{llm_description}'."
            else:
                prompt_line += "."
            text_parts.append(prompt_line)

        elif mode == 'header_mode':
            prompt_line = f"The attribute with header: '{header}'"
            if llm_description:
                prompt_line += f" is described as: '{llm_description}'."
            else:
                prompt_line += "."
            text_parts.append(prompt_line)

        # Add value examples if available
        if attribute_examples:
            text_parts.append(f"Examples of values for this attribute include: '{attribute_examples}'.")

        # Add column statistics if available
        if num_values > 0:
            text_parts.append(f"The dataset column for '{header}' contains {num_values} entries.")

        if max_length > 0 and min_length > 0:
            text_parts.append("Key statistics for the column:")
            text_parts.append(f"- Maximum value length: {max_length} characters.")
            text_parts.append(f"- Minimum value length: {min_length} characters.")
            text_parts.append(f"- Average value length: {avg_length:.1f} characters.")

        # Add most frequent values if available
        if most_common_values:
            text_parts.append("Top 20 most frequent values in the column:")
            text_parts.append(", ".join([str(v) for v in most_common_values[:20]]))

        # Join and truncate the final prompt
        final_prompt = "\n".join([part for part in text_parts if part])

        if final_prompt:
            return self._truncate_text(final_prompt, max_tokens)
        else:
            return None

    def _get_embedding(self, text: str, model: str) -> Union[list[float], None]:
        """
        Private helper method to get the embedding for a given text using the specified model.

        Args:
            text (str): The text for which to generate an embedding.
            model (str): The name of the OpenAI embedding model to use.

        Returns:
            Union[list[float], None]: The embedding for the text, or None if an error occurs.
        """
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(add_timestamp(color_text(f"Error getting embedding for text: '{text[:50]}...': {e}", "red")))
            return None

    def generate_embeddings(self,
        data_lake_list: Union[list[dict], str],
        prompt_mode: str = 'semantic_mode',
        embedding_model: str = 'text-embedding-3-small',
        output_directory: str | None = None
    ) -> list[dict]:
        """
        Generates a list of embeddings for each column in the data lake representation.

        Args:
            data_lake_list (list[dict] | str): The list of dictionaries representing the data lake,
                                            as generated by DataLoader.load_and_describe_datalake.
            prompt_mode (str): One of 'semantic_mode' or 'data_profiling_mode'. See _prepare_embedding_prompt
                                            for more information.
            embedding_model (str): The name of the OpenAI embedding model to use.
            output_directory (str | None): If specified, the generated embeddings will be saved to a file
                                        in this directory, named "embeddings.json".

        Returns:
            list[dict]: A list of dictionaries, each containing the embedding for a column and its file name,
                        as well as the semantic annotation for the column (if available).
        """
        if not data_lake_list:
            print(add_timestamp(color_text("No data lake information provided for schema pruning.", "yellow")))
            return None
            
        if isinstance(data_lake_list, str):
            if not os.path.exists(data_lake_list):
                print(add_timestamp(color_text(f"Error: Data lake JSON file not found at {data_lake_list}", "red")))
            data_lake_list = load_datalake_json(data_lake_list)
        
        embeddings_list = []
        max_prompt_tokens = 8191 if 'text-embedding-3-large' in embedding_model else 2048

        total_columns = sum(len(file_info['columns']) for file_info in data_lake_list)

        with tqdm(total=total_columns, desc=add_timestamp("Generating Embeddings")) as pbar:
            for file_info in data_lake_list:
                for column_dict in file_info['columns']:
                    full_column_name = f"{file_info['file_name']}:{column_dict['column_name']}"

                    prompt_text = self._prepare_embedding_prompt(
                        column_data=column_dict,
                        mode=prompt_mode,
                        max_tokens=max_prompt_tokens
                    )

                    if prompt_text is None:
                        pbar.update(1)
                        continue

                    embedding = self._get_embedding(prompt_text, model=embedding_model)

                    if embedding is not None:
                        embeddings_list.append({
                            'column_name': full_column_name,
                            'embedding': embedding,
                            'semantic_annotation': column_dict.get('semantic_annotation', 'NA')
                        })

                    pbar.update(1)
                    time.sleep(0.1)

        print(add_timestamp(color_text(f"Finished generating embeddings for {len(embeddings_list)} columns.", "green")))
        
        if output_directory:
            try:
                create_directory_if_not_exists(output_directory)
                output_file = os.path.join(output_directory, "embeddings.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_list, f, indent=2)
                print(add_timestamp(color_text(f"Embeddings saved to: {output_file}", "green")))
            except Exception as e:
                print(add_timestamp(color_text(f"Error saving embeddings to {output_file}: {e}", "red")))
        
        return embeddings_list

    def _compute_distances(self,
        embeddings: list[dict],
        output_directory
    ) -> pd.DataFrame:
        """
        Compute cosine similarity, euclidean distance, and ANNS distances between all pairs of column embeddings.

        Args:
            embeddings (list[dict]): A list of dictionaries, each containing the embedding for a column and its file name,
                                     as well as the semantic annotation for the column (if available).
            output_directory (str): The directory where the output files will be saved.

        Returns:
            pd.DataFrame: A DataFrame containing the distances and similarities between all pairs of columns.
        """
        column_names = [d['column_name'] for d in embeddings]
        embedding_matrix = np.array([d['embedding'] for d in embeddings])

        # --- 1. Calculate Cosine and Euclidean Distances ---
        print(add_timestamp(color_text("Calculating cosine similarity and euclidean distances...", 'cyan')))
        cosine_sim_matrix = cosine_similarity(embedding_matrix)
        euclidean_dist_matrix = euclidean_distances(embedding_matrix)

        # --- 2. Calculate ANNS Distances (using FAISS) ---
        print(add_timestamp(color_text("Calculating ANNS distances...", 'cyan')))
        try:
            d = embedding_matrix.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embedding_matrix.astype('float32'))
            distances_anns, indices_anns = index.search(embedding_matrix.astype('float32'), len(column_names))

        except ImportError:
            print(add_timestamp(color_text("FAISS library not found. ANNS distances will be skipped.", "red")))
            distances_anns, indices_anns = None, None
        except Exception as e:
            print(add_timestamp(color_text(f"Error calculating ANNS distances: {e}. ANNS distances will be skipped.", "red")))
            distances_anns, indices_anns = None, None
        
        # Export distances
        distances = []
        for i in range(len(column_names)):
            for j in range(len(column_names)):
                if i != j: # Exclude self-comparison
                    distances.append({
                        "column_1": column_names[i],
                        "column_2": column_names[j],
                        "cosine_similarity": cosine_sim_matrix[i, j],
                        "euclidean_distance": euclidean_dist_matrix[i, j],
                        "distance_anns": distances_anns[i, j] if distances_anns is not None else None
                    })
        sorted_distances = sorted(distances, key=lambda x: (x['euclidean_distance'], -x['cosine_similarity'], x['distance_anns']))
        
        if output_directory:
            try:
                create_directory_if_not_exists(output_directory)
                output_file = os.path.join(output_directory, "distances.csv")
                df_distances = pd.DataFrame(sorted_distances)
                df_distances.to_csv(output_file, index=False, encoding='utf-8')
                print(add_timestamp(color_text(f"Distances saved to: {output_file}", "green")))
            except Exception as e:
                print(add_timestamp(color_text(f"Error saving distances to {output_file}: {e}", "red")))
        
        return df_distances

    def _prepare_nodes(self,
        embeddings: list[dict],
        output_directory: str | None = None
    ) -> pd.DataFrame:
        """
        Prepare nodes DataFrame for export to Neo4j. This function takes the input embeddings,
        drops the "semantic_annotation" column, splits the "column_name" column into "docID" and "colname",
        removes duplicates by "docID" and "colname", pivots the DataFrame to have one row per file
        and one column per CSV column name, cleans the column names, and renames the "docID" column
        to the Neo4j-compatible node ID. The resulting DataFrame is then saved to a CSV file in
        the specified output directory.

        Args:
            embeddings (list[dict]): The list of dictionaries with 'column_name', 'embedding', and 'semantic_annotation'.
            output_directory (str): The directory where output files will be saved.

        Returns:
            pd.DataFrame: The prepared nodes DataFrame.
        """
        print(add_timestamp(color_text("Generating Neo4j nodes CSV...", 'cyan')))
        nodes = embeddings.copy()
        nodes = pd.DataFrame(nodes)
        nodes = nodes.drop(columns=["semantic_annotation"])
        nodes[["docID", "colname"]] = nodes["column_name"].str.split(":", expand=True) # Split 'column_name' into 'docID' (filename) and 'colname' (column header)
        nodes = nodes.drop(columns=["column_name"])
        nodes = nodes.drop_duplicates(subset=["docID", "colname"]) # Remove duplicates by 'docID' and 'colname'
        pivoted = nodes.pivot(index="docID", columns="colname", values="embedding").reset_index() # Pivot: one row per file, one column per CSV column name
        pivoted.columns = pivoted.columns.str.replace(":", "_", regex=False) # Clean column names: replace ':' with '_' and empty names with '_'
        pivoted.columns = pivoted.columns.str.replace("^$", "_", regex=True)
        pivoted = pivoted.rename(columns={"docID": "docID:ID"}) # Rename 'docID' to Neo4j-compatible node ID
        
        if output_directory:
            try:
                create_directory_if_not_exists(output_directory)
                output_file = os.path.join(output_directory, "nodes.csv")
                pivoted.to_csv(output_file, index=False, encoding='utf-8')
                print(add_timestamp(color_text(f"Neo4j nodes saved to: {output_file}", "green")))
            except Exception as e:
                print(add_timestamp(color_text(f"Error saving neo4j nodes to {output_file}: {e}", "red")))
        
        return pivoted
    
    def _prepare_edges(self,
        df: pd.DataFrame,
        cosine_sim_threshold: float = 0.5,
        anns_threshold: float = 0.2,
        output_directory: str | None = None
    ) -> pd.DataFrame:    
        """
        Prepare edges DataFrame for export to Neo4j. This function takes the input DataFrame, 
        splits the "column_1" and "column_2" columns into "docID" and "colname", inverts the 
        direction of the edges and removes duplicates, cleans the column names, and renames the 
        "docID" column to the Neo4j-compatible node ID. The resulting DataFrame is then saved to 
        a CSV file in the specified output directory.

        Args:
            df (pd.DataFrame): The input DataFrame with 'column_1', 'column_2', 'cosine_similarity', and 'distance_anns'.
            output_directory (str): The directory where output files will be saved.

        Returns:
            pd.DataFrame: The prepared edges DataFrame.
        """
        df[['f1', 'c1']] = df['column_1'].str.split(':', expand=True)
        df[['f2', 'c2']] = df['column_2'].str.split(':', expand=True)
        df['c1'] = df['c1'].str.replace(":", "_", regex=False).str.replace("^$", "_", regex=True)
        df['c2'] = df['c2'].str.replace(":", "_", regex=False).str.replace("^$", "_", regex=True)
        df[':TYPE'] = df['c1'] + " -> " + df['c2']
        df = df.rename(columns={'f1': ':START_ID', 'f2': ':END_ID'})
        df = df.drop(columns=['column_1', 'column_2', 'c1', 'c2'])
        
        def invert(df):
            """
            Inverts the direction of the edges in the given DataFrame.
            """
            inv = df.copy()
            inv = inv.rename(columns={':START_ID': ':END_ID', ':END_ID': ':START_ID'})
            inv[':TYPE'] = inv[':TYPE'].apply(lambda x: ' -> '.join(x.split(' -> ')[::-1]))
            return inv
        
        df_all = pd.concat([df, invert(df)], ignore_index=True).drop_duplicates()
        
        edges = df_all[
            (df_all[':START_ID'] != df_all[':END_ID']) &
            (df_all['cosine_similarity'] >= cosine_sim_threshold) &
            (df_all['distance_anns'] <= anns_threshold)
        ].copy()

        edges['pair_type'] = edges.apply(
            lambda r: (tuple(sorted([r[':START_ID'], r[':END_ID']])),
                ' -> '.join(sorted(r[':TYPE'].split(' -> ')))
                if r[':START_ID'] > r[':END_ID'] else r[':TYPE']),
            axis=1)
        edges =  edges.drop_duplicates('pair_type').drop(columns=['pair_type'])
        
        if output_directory:
            try:
                create_directory_if_not_exists(output_directory)
                output_file = os.path.join(output_directory, "edges.csv")
                edges.to_csv(output_file, index=False, encoding='utf-8')
                print(add_timestamp(color_text(f"Neo4j edges saved to: {output_file}", "green")))
            except Exception as e:
                print(add_timestamp(color_text(f"Error saving neo4j edges to {output_file}: {e}", "red")))
        
        return edges

    def compute_distances_and_export_neo4j(self,
        embeddings: Union[list[dict], str],
        cosine_sim_threshold: float = 0.5,
        anns_threshold: float = 0.2,
        output_directory: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes cosine similarity, euclidean distance, and ANNS distances between all pairs of column embeddings, 
        and exports the results as CSV files for Neo4j nodes and edges.

        Args:
            embeddings: A list of dictionaries, each containing the embedding for a column and its file name,
                        as well as the semantic annotation for the column (if available). Alternatively, a string
                        path to a JSON file containing the embeddings.
            cosine_sim_threshold: The minimum cosine similarity required for a pair of columns to be considered joinable.
            anns_threshold: The maximum ANNS distance required for a pair of columns to be considered joinable.
            output_directory: The directory where the output CSV files will be saved.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The computed distances DataFrame, the nodes DataFrame,
                and the edges DataFrame.
        """
        if not embeddings:
            print(add_timestamp(color_text("No embeddings provided. Skipping distance calculation and export.", "yellow")))
            return None
            
        if isinstance(embeddings, str):
            if not os.path.exists(embeddings):
                print(add_timestamp(color_text(f"Error: embeddings JSON file not found at {embeddings}", "red")))
            with open(embeddings, "r") as f:
                embeddings = json.load(f)

        # --- Compute Distances ---
        df_distances = self._compute_distances(embeddings, output_directory)

        # --- Prepare Neo4j Nodes CSV ---
        df_nodes = self._prepare_nodes(embeddings, output_directory)
        
        # --- Prepare Neo4j Edges CSV ---
        df_edges = self._prepare_edges(df_distances, cosine_sim_threshold, anns_threshold, output_directory)
        
        return df_distances, df_nodes, df_edges

