from Embedder import Embedder
import time
import os
import glob
import json

def load_matches(file_path):
    """Load matches from a JSON file and return them as a set of tuples."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {
        (
            m["source_table"], 
            m["source_column"], 
            m["target_table"], 
            m["target_column"]
        )
        for m in data["matches"]
    }

def calculate_metrics(ground_truth_path, suggested_path):
    """Calculate precision, recall, and F1 score."""
    # Load matches
    gt_matches = load_matches(ground_truth_path)
    suggested_matches = load_matches(suggested_path)

    # Calculate metrics
    tp = len(gt_matches & suggested_matches)  # True Positives (intersection)
    fp = len(suggested_matches - gt_matches)  # False Positives
    fn = len(gt_matches - suggested_matches)  # False Negatives

    # Handle division by zero cases
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }

dataset = ... # Path of a dataset from the Valentine collectio e.g.n "/Documents/Datasets/Valentine/Wikidata/Musicians/Musicians_joinable"
output_folder_name = os.path.join("PROVA")
output_folder = os.path.join("outputs", output_folder_name)
methods = ["semantic_annotation"]
                
dir = dataset
for method in methods:
    
    try:
        mapping_filename_pattern = os.path.join(dir, f"*_mapping.json")
        ground_truth_file = glob.glob(mapping_filename_pattern)[0]

        time_start = time.time()

        embedder = Embedder(dir, method, output_folder, False)
        embedder.makeSchema()
        embedder.pruneSchema()
        embedder.annotateDatasets()
        embeddings = embedder.embedDatasets()
        embedder.computeDistances(embeddings)
        embedder.saveGraph()

        time_end = time.time()
        print(f"Execution time: {time_end - time_start} seconds")

        metrics = calculate_metrics(ground_truth_file, embedder.mappings_file)
        
        metrics_filename = f"{os.path.basename(dir)}.json"
        metrics_output_folder = os.path.join(output_folder, "metrics", method)
        
        metrics_dir = os.path.join(metrics_output_folder, metrics_filename)
        os.makedirs(metrics_output_folder, exist_ok = True)
        
        with open(metrics_dir, "w") as file:
            json.dump(metrics, file, indent = 4)
    except Exception as e:
        log_file_path = os.path.join("outputs", "log.txt")
        with open(log_file_path, "a") as file:
            file.write("\n")
            file.write(dir)