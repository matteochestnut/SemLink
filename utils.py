import csv
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import faiss

from colorama import Fore, Style
import time
import os

import tiktoken

import csv
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS'] = '1'

def color_text(text, color='white'):
    """
    Colors text using colorama colors.
    Args:
        text (str): Text to be colored
        color (str): Color name to use, defaults to white. 
                    Valid colors: black, red, green, yellow, blue, magenta, cyan, white
    Returns:
        str: Colored text string with reset style appended
    """
    colors = {
        'black': '\033[30m',
        'red': '\033[31m', 
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m'
    }
    color_code = colors.get(color.lower(), colors['white'])
    return f"{color_code}{text}{Style.RESET_ALL}"

def add_timestamp(text):
    """
    Adds a timestamp before the provided text.
    Args:
        text (str): Text to prepend timestamp to
    Returns:
        str: Text with timestamp prepended in format [HH:MM:SS]
    """
    timestamp = time.strftime("[%H:%M:%S]")
    return f"{timestamp} {text}"

def count_tokens(text, model_name='text-embedding-ada-002'):
    """Conta il numero di token nel testo usando tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def truncate_text(text, max_tokens, model_name='text-embedding-ada-002'):
    """Tronca il testo per rientrare nel limite dei token."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    encoded_text = encoding.encode(text)
    truncated_text = encoding.decode(encoded_text[:max_tokens])
    return truncated_text









































def save_embeddings_pickle(embeddings, method_name, ts_run):
    # Save embeddings in a pickle file
    pickle_output_dir = f"./embedder/saved_embeddings/pickle/{ts_run}"
    pickle_output_dir = os.path.join(pickle_output_dir, "openai")
    
    #print(f'Directory: {pickle_output_dir}')
    
    os.makedirs(pickle_output_dir, exist_ok=True)
    pickle_output_file = f"{pickle_output_dir}/embeddings_{method_name}.pkl"
    with open(pickle_output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(add_timestamp(color_text(f"Embeddings saved in: {pickle_output_file}", 'green')))

def calculate_similarities(embeddings):
    """
    Calculate cosine similarity and euclidean distance.
    """
    X = np.array([item['embedding'] for item in embeddings])
    column_names = [item['column_name'] for item in embeddings]

    # Calculate cosine similarity and euclidean distance
    cosine_sim_matrix = cosine_similarity(X)
    euclidean_dist_matrix = euclidean_distances(X)

    return cosine_sim_matrix, euclidean_dist_matrix, column_names

def get_euclidean_cosine_distances(embeddings, cosine_sim_matrix, euclidean_dist_matrix, method_name, ts_run):
    """
    Prints the distances and cosine similarities in order, with the reference files.
    Also saves the distance results in a CSV file.

    Args:
        embeddings (list): List of embeddings with column names.
        cosine_sim_matrix (ndarray): Cosine similarity matrix.
        euclidean_dist_matrix (ndarray): Euclidean distance matrix.
        method_name (str): Name of the method used.
    """
    column_names = [item['column_name'] for item in embeddings]
    
    # Create a list of all pairs combinations, including both directions
    distances = []
    for i in range(len(column_names)):
        for j in range(len(column_names)):  # Include both directions
            if i != j:  # Exclude self-comparison
                distances.append({
                    "column_1": column_names[i],
                    "column_2": column_names[j],
                    "cosine_similarity": cosine_sim_matrix[i, j],
                    "euclidean_distance": euclidean_dist_matrix[i, j]
                })
    
    # Sort by euclidean distance ascending and, in case of equality, by cosine similarity descending
    sorted_distances = sorted(distances, key=lambda x: (x['euclidean_distance'], -x['cosine_similarity']))
    
    csv_output_dir = f"./embedder/saved_embeddings/csv/OTHER/openai/{ts_run}"
    #print(add_timestamp(color_text(f'Directory csv: {csv_output_dir}', 'white')))
    
    # Directory creation and save results
    os.makedirs(csv_output_dir, exist_ok=True)
    csv_output_file = f"{csv_output_dir}/distances_{method_name}.csv"
    df_output = pd.DataFrame(sorted_distances)
    df_output.to_csv(csv_output_file, index=False, encoding='utf-8')
    print(add_timestamp(color_text(f"Euclid. and Cosine distances saved in: {csv_output_file}", 'green')))

def get_anns_distances(embeddings, ts_run):
    """
    Calculates distances using ANNS and returns all combinations except self-comparisons.
    """
    try:
        # Extract embeddings and column names
        X = np.array([item['embedding'] for item in embeddings], dtype=np.float32)  # Ensure float32
        column_names = [item['column_name'] for item in embeddings]

        # Create FAISS index with explicit thread control
        dimension = X.shape[1]
        faiss.omp_set_num_threads(1)  # Limit OpenMP threads
        index = faiss.IndexFlatL2(dimension)
        
        # not needed as of now
        # Use GPU if available
        # try:
        #     res = faiss.StandardGpuResources()
        #     index = faiss.index_cpu_to_gpu(res, 0, index)
        # except Exception:
        #     pass  # Fall back to CPU if GPU not available
        
        index.add(X)

        # Find all distances between columns
        total_columns = len(column_names)
        distances, indices = index.search(X, total_columns)

        results = []
        for i in range(len(column_names)):
            for j in range(total_columns):
                if i != indices[i, j]:
                    results.append({
                        "column_1": column_names[i],
                        "column_2": column_names[indices[i, j]],
                        "distance": float(distances[i, j])  # Convert to Python float
                    })

        sorted_results = sorted(results, key=lambda x: x['distance'])

        # Save results
        csv_output_dir = f"./embedder/saved_embeddings/csv/ANNS/openai/{ts_run}"
        os.makedirs(csv_output_dir, exist_ok=True)
        csv_output_file = f"{csv_output_dir}/anns_distances.csv"
        df_output = pd.DataFrame(sorted_results)
        df_output.to_csv(csv_output_file, index=False, encoding='utf-8')
        print(add_timestamp(color_text(f"ANNS distances saved in: {csv_output_file}", 'green')))

        return sorted_results

    finally:
        # Cleanup
        del index
        import gc
        gc.collect()








def get_table_metadata(csv_dir):
    """
    Scans the directory for CSV files and returns a list of dictionaries
    with each table's file name, size (in bytes), and number of columns.
    """
    tables = []
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv') or filename.endswith('.tsv'):
            path = os.path.join(csv_dir, filename)
            size = os.stat(path).st_size
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    num_columns = len(header)
                except StopIteration:
                    num_columns = 0  # empty file
            tables.append({
                'filename': filename,
                'size': size,
                'num_columns': num_columns
            })
    return tables

def select_subset(tables, n):
    """ Togliere l'hard coded di prendere 10 tabelle per forza """
    # Sort tables by different criteria
    tables_by_size = sorted(tables, key=lambda x: x['size'])
    tables_by_columns = sorted(tables, key=lambda x: x['num_columns'])
    
    # 10 smallest and 10 largest tables by size
    smallest_size = [t['filename'] for t in tables_by_size[:5]]
    largest_size = [t['filename'] for t in tables_by_size[-5:]]
    
    # 10 tables with smallest and 10 tables with largest number of columns
    smallest_cols = [t['filename'] for t in tables_by_columns[:5]]
    largest_cols = [t['filename'] for t in tables_by_columns[-5:]]
    
    # Create the candidate set (ensuring uniqueness)
    selected = set(smallest_size + largest_size + smallest_cols + largest_cols)
    
    # Determine how many additional tables are needed
    additional_needed = n - len(selected)
    
    # Create a set of filenames from all tables
    all_files = set(t['filename'] for t in tables)
    remaining = list(all_files - selected)
    
    # Randomly sample the remaining tables
    if additional_needed > len(remaining):
        print("Warning: Not enough tables to sample from. Returning all available tables.")
        additional = remaining
    else:
        additional = random.sample(remaining, additional_needed)
    
    # Final set of 100 unique tables
    final_tables = list(selected) + additional
    return final_tables