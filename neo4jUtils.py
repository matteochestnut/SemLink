import os
import pandas as pd
import numpy as np
from utils import color_text, add_timestamp
import json

def get_stats(df):
    if 'cosine_similarity' in df.columns:
        mean = df['cosine_similarity'].mean()
        median = df['cosine_similarity'].median()
        mean_same = df[df[':TYPE'].str.split(' -> ').apply(lambda x: x[0] == x[1])]['cosine_similarity'].mean()
        median_same = df[df[':TYPE'].str.split(' -> ').apply(lambda x: x[0] == x[1])]['cosine_similarity'].median()
        return mean, median, mean_same, median_same
    else:
        mean = df['distance_anns'].mean()
        median = df['distance_anns'].median()
        mean_same = df[df[':TYPE'].str.split(' -> ').apply(lambda x: x[0] == x[1])]['distance_anns'].mean()
        median_same = df[df[':TYPE'].str.split(' -> ').apply(lambda x: x[0] == x[1])]['distance_anns'].median()
        
        return mean, median, mean_same, median_same

def prepare_nodes(embedding_file):

    # Load embeddings
    embeddings = pd.DataFrame(pd.read_pickle(embedding_file))
    embeddings[['column_csv', 'column_col']] = embeddings['column_name'].str.split(':', expand=True, n = 1)
    embeddings = embeddings.drop(columns=["column_name"])
    
    # Group by column_csv and aggregate the embeddings
    unique_columns = embeddings["column_col"].unique()
    new_columns = {
        col: np.where(embeddings["column_col"] == col, embeddings["embedding"], None)
        for col in unique_columns
    }

    # Concatenate the embeddings and the new columns
    embeddings = pd.concat(
        [embeddings[["column_csv", "embedding"]], pd.DataFrame(new_columns)], axis=1
    ).drop(columns=["embedding"])

    # Group by column_csv and aggregate the embeddings
    embeddings = embeddings.groupby("column_csv").agg(
        {col: lambda x: next((v for v in x if v is not None), None) for col in unique_columns}
    ).reset_index()

    # Replace : with _ and empty column headers with _
    embeddings.columns = embeddings.columns.str.replace(":", "_") # : is not allowed in Neo4j attributes' labels (next line you see why)
    embeddings.columns = embeddings.columns.str.replace("^$", "_", regex=True) # empty column header

    # Rename column_csv to docID:ID
    embeddings = embeddings.rename(columns={"column_csv": "docID:ID"})

    # Add :LABEL attribute
    embeddings[":LABEL"] = "CSV" # We only have CSV for now

    # Save the embeddings to a CSV file
    return embeddings

def prepare_edges(distances_file, anns_distances_file):
    # Load cosine similarities
    cos_sim_df = pd.read_csv(distances_file)
    cos_sim = cos_sim_df.copy()

    cos_sim[['column_1_csv', 'column_1_col']] = cos_sim['column_1'].str.split(':', expand=True, n = 1)
    cos_sim[['column_2_csv', 'column_2_col']] = cos_sim['column_2'].str.split(':', expand=True, n = 1)

    # Replace : with _ and empty column headers with _
    cos_sim["column_1_col"] = cos_sim["column_1_col"].str.replace(":", "_")
    cos_sim["column_1_col"] = cos_sim["column_1_col"].str.replace("^$", "_", regex=True)
    cos_sim["column_2_col"] = cos_sim["column_2_col"].str.replace(":", "_")
    cos_sim["column_2_col"] = cos_sim["column_2_col"].str.replace("^$", "_", regex=True)

    # Add :TYPE attribute
    cos_sim[':TYPE'] = cos_sim['column_1_col'] + " -> " + cos_sim['column_2_col']
    cos_sim = cos_sim.drop(columns=["column_1", "column_2", "column_1_col", "column_2_col"])

    # Load ANN distances
    anns_df = pd.read_csv(anns_distances_file)
    anns = anns_df.copy()
    anns = anns.rename(columns={"distance": "distance_anns"})
    anns[['column_1_csv', 'column_1_col']] = anns['column_1'].str.split(':', expand=True, n = 1)
    anns[['column_2_csv', 'column_2_col']] = anns['column_2'].str.split(':', expand=True, n = 1)

    # Replace : with _ and empty column headers with _
    anns["column_1_col"] = anns["column_1_col"].str.replace(":", "_")
    anns["column_1_col"] = anns["column_1_col"].str.replace("^$", "_", regex=True)
    anns["column_2_col"] = anns["column_2_col"].str.replace(":", "_")
    anns["column_2_col"] = anns["column_2_col"].str.replace("^$", "_", regex=True)

    # Add :TYPE attribute
    anns[':TYPE'] = anns['column_1_col'] + " -> " + anns['column_2_col']
    anns = anns.drop(columns=["column_1", "column_2", "column_1_col", "column_2_col"])
    
    # Load cosine similarities inverse
    cos_sim_inverse = cos_sim_df.copy()
    cos_sim_inverse[['column_1_csv', 'column_1_col']] = cos_sim_inverse['column_1'].str.split(':', expand=True, n = 1)
    cos_sim_inverse[['column_2_csv', 'column_2_col']] = cos_sim_inverse['column_2'].str.split(':', expand=True, n = 1)

    # Replace : with _ and empty column headers with _
    cos_sim_inverse["column_1_col"] = cos_sim_inverse["column_1_col"].str.replace(":", "_")
    cos_sim_inverse["column_1_col"] = cos_sim_inverse["column_1_col"].str.replace("^$", "_", regex=True)
    cos_sim_inverse["column_2_col"] = cos_sim_inverse["column_2_col"].str.replace(":", "_")
    cos_sim_inverse["column_2_col"] = cos_sim_inverse["column_2_col"].str.replace("^$", "_", regex=True)

    # Rename columns
    cos_sim_inverse.rename(columns={"column_1_csv": "column_2_csv", "column_1_col": "column_2_col",
                            "column_2_csv": "column_1_csv", "column_2_col": "column_1_col"}, inplace=True)

    # Add :TYPE attribute
    cos_sim_inverse[':TYPE'] = cos_sim_inverse['column_1_col'] + " -> " + cos_sim_inverse['column_2_col']
    cos_sim_inverse = cos_sim_inverse.drop(columns=["column_1", "column_2", "column_1_col", "column_2_col"])

    # Concatenate cosine similarities and cosine similarities inverse
    cos_sim = pd.concat([cos_sim, cos_sim_inverse]).drop_duplicates()
    cos_sim = cos_sim.rename(columns={"column_1_csv": ":START_ID", "column_2_csv": ":END_ID"})

    # Load ANN distances inverse    
    anns_inverse = anns_df.copy()
    anns_inverse[['column_1_csv', 'column_1_col']] = anns_inverse['column_1'].str.split(':', expand=True, n = 1)
    anns_inverse[['column_2_csv', 'column_2_col']] = anns_inverse['column_2'].str.split(':', expand=True, n = 1)
    anns_inverse = anns_inverse.rename(columns={"distance": "distance_anns"})

    # Replace : with _ and empty column headers with _
    anns_inverse["column_1_col"] = anns_inverse["column_1_col"].str.replace(":", "_")
    anns_inverse["column_1_col"] = anns_inverse["column_1_col"].str.replace("^$", "_", regex=True)
    anns_inverse["column_2_col"] = anns_inverse["column_2_col"].str.replace(":", "_")
    anns_inverse["column_2_col"] = anns_inverse["column_2_col"].str.replace("^$", "_", regex=True)

    # Rename columns
    anns_inverse.rename(columns={"column_1_csv": "column_2_csv", "column_1_col": "column_2_col",
                            "column_2_csv": "column_1_csv", "column_2_col": "column_1_col"}, inplace=True)

    # Add :TYPE attribute
    anns_inverse[':TYPE'] = anns_inverse['column_1_col'] + " -> " + anns_inverse['column_2_col']
    anns_inverse = anns_inverse.drop(columns=["column_1", "column_2", "column_1_col", "column_2_col"])

    anns = pd.concat([anns, anns_inverse]).drop_duplicates()
    anns = anns.rename(columns={"column_1_csv": ":START_ID", "column_2_csv": ":END_ID"})
    # Get stats
    mean_cos, median_cos, mean_same_cos, median_same_cos = get_stats(cos_sim)
    print(add_timestamp(color_text(f"Cosine similarity mean: {mean_cos}", "yellow")))
    print(add_timestamp(color_text(f"Cosine similarity median: {median_cos}", "yellow")))
    print(add_timestamp(color_text(f"Cosine similarity mean same annotation: {mean_same_cos}", "yellow")))
    print(add_timestamp(color_text(f"Cosine similarity median same annotation: {median_same_cos}", "yellow")))

    mean_anns, median_anns, mean_same_anns, median_same_anns = get_stats(anns)
    print(add_timestamp(color_text(f"ANNs mean: {mean_anns}", "yellow")))
    print(add_timestamp(color_text(f"ANNs median: {median_anns}", "yellow")))
    print(add_timestamp(color_text(f"ANNs mean same annotation: {mean_same_anns}", "yellow")))
    print(add_timestamp(color_text(f"ANNs median same annotation: {median_same_anns}", "yellow")))

    stats = {
        "cosine_similarity": {
            "mean": mean_cos,
            "median": median_cos,
            "mean_same_annotation": mean_same_cos,
            "median_same_annotation": median_same_cos
        },
        "anns": {
            "mean": mean_anns,
            "median": median_anns,
            "mean_same_annotation": mean_same_anns,
            "median_same_annotation": median_same_anns
        }
    }

    # Get relationships
    relationships = pd.merge(cos_sim, anns, on=[":START_ID", ":END_ID", ":TYPE"])
    
    # Filter relationships using thresholds
    relationships = relationships[relationships[':START_ID']!=relationships[':END_ID']]
    relationships = relationships[(relationships['distance_anns'] <= 0.2) & (relationships['cosine_similarity'] >= 0.5)]

    def create_relationship_key(row):
        """Create a unique key that considers both node order and relationship type"""
        nodes = sorted([row[':START_ID'], row[':END_ID']])
        types = row[':TYPE'].split(' -> ')
        if nodes[0] != row[':START_ID']:
            types = types[::-1]
        return (tuple(nodes), ' -> '.join(types))

    if not relationships.empty:
        # Create a unique key that considers both node order and relationship direction
        relationships['pair_type'] = relationships.apply(create_relationship_key, axis=1)

        # Ensure unidirectionality by keeping only one pair
        relationships = relationships.drop_duplicates(subset=['pair_type']).drop(columns=['pair_type'])
    
    return stats, relationships


def get_mappings(df):

    # Check if the DataFrame is empty
    if df.empty:
        result = {"matches": []}
    else:
        # Determine source and target columns based on the first entry
        first_row = df.iloc[0]
        start_id = first_row[':START_ID']
        end_id = first_row[':END_ID']

        source_table_col = None
        target_table_col = None

        # Check for '_source' in either START_ID or END_ID
        if '_source' in start_id:
            source_table_col = ':START_ID'
            source_id = 0
        elif '_source' in end_id:
            source_table_col = ':END_ID'
            source_id = 1

        # Check for '_target' in either START_ID or END_ID
        if '_target' in start_id:
            target_table_col = ':START_ID'
            target_id = 0
        elif '_target' in end_id:
            target_table_col = ':END_ID'
            target_id = 1

        # Fallback if not detected (though not needed with given data)
        if source_table_col is None:
            source_table_col = ':START_ID'
        if target_table_col is None:
            target_table_col = ':END_ID'

        pd_dataframe = []
        # Prepare the matches list
        matches = []
        for _, row in df.iterrows():
            # Extract source and target tables, removing .csv extension
            source_table = row[source_table_col].replace('.csv', '')
            target_table = row[target_table_col].replace('.csv', '')
            source_table = source_table.replace('.tsv', '')
            target_table = target_table.replace('.tsv', '')

            # Split the TYPE into source and target columns
            type_parts = row[':TYPE'].split('->')
            source_column = type_parts[source_id].strip()
            target_column = type_parts[target_id].strip()

            # Append the dictionary to matches
            matches.append({
                "source_table": source_table,
                "source_column": source_column,
                "target_table": target_table,
                "target_column": target_column
            })
            
            row = [source_table, target_table, source_column, target_column]
            pd_dataframe.append(row)

        result = {"matches": matches}
        
        df = pd.DataFrame(pd_dataframe, columns=["query_table", "candidate_table", "query_column", "candidate_column"]) # Create DataFrame

    return result, df