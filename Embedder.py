from tqdm import tqdm
import os
from dotenv import load_dotenv
import time
import openai
import json
import glob
import sys
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import faiss

from preprocessUtils import process_csv_file, find_latest_processor_run, collect_all_column_data, unify_column_data, generate_linkml_schema, process_csv_json_pair
from utils import add_timestamp, color_text, count_tokens, truncate_text, select_subset, get_table_metadata
from yamlParser import yaml_parser
from pruningUtils import process_dataset, classify_columns
from embeddingUtils import load_datasets, get_embedding, prepare_text_with_semantic_annotation
from neo4jUtils import prepare_nodes, prepare_edges, get_mappings

current_dir = os.path.dirname(os.path.abspath(__file__))
archetype_path = os.path.join(current_dir, 'archetype')
sys.path.insert(1, archetype_path)
from archetype.src.predict import ArcheTypePredictor

os.system('clear')

class Embedder:
    def __init__(self, dataset_dir: str, method: str, output_folder: str, eurostat: bool) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_folder = os.path.basename(dataset_dir)
        self.method = method
        self.output_folder = output_folder
        self.schema_file = None
        self.pruned_schema_file = None
        self.annotations_file = None
        self.embedding_file = None
        self.embeddings = None
        self.distances_file = None
        self.distances = None
        self.anns_distances_file = None
        self.anns_distances = None
        self.nodes_file = None
        self.edges_file = None
        self.mappings_file = None
        self.mappings = None
        self.eurostat = eurostat # if true process json metadata
        
        self.file_blacklist = []
        
        # Load environment variables and setting up OpenAi clietn
        print("Loading environment...")
        print("Loading OpenAi key...")
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key is None:
            print(add_timestamp(color_text("OpenAI key not found", "red")))
            raise ValueError("OpenAI key not found")
        else:
            print(add_timestamp(color_text("OpenAI key loaded", "green")))
            openai.api_key = openai_key
        
        self.client = openai.OpenAI()
        
        # Set up run timestamp identifier
        self.ts_run = time.strftime("run_"+"%Y%m%d_%H%M%S")
    
    def setBlacklistedFiles(self, blacklisted_files: list) -> None:
        self.file_blacklist = blacklisted_files
    
    def makeSchema(self, entries_to_process = 1000, subset_for_schema = True) -> None:
        
        if self.method == "semantic_annotation":
            """Function to process all CSV/TSV files"""
            # Check if dataset directory exists
            if not os.path.exists(self.dataset_dir):
                print(f"Error: Directory {self.dataset_dir} not found")
                return
            print(f"Processing files from {self.dataset_dir}")
            
            # Create preprocess output directory
            output_dir = os.path.join(self.output_folder, self.dataset_folder, self.method, "preprocess", self.ts_run)
            os.makedirs(output_dir, exist_ok = True)
            print(f"Output will be saved to {output_dir}")
                
            # Get all CSV and TSV files
            data_files = [f for f in os.listdir(self.dataset_dir) if f.lower().endswith(('.csv', '.tsv'))]
            print(f"Found {len(data_files)} data files")
            
            if subset_for_schema and len(data_files) > 50:
                tables = get_table_metadata(self.dataset_dir)
                data_files = select_subset(tables)

                print(f"Processing {len(data_files)} data files")
            
            results = []
            for i in range(len(data_files)): # magari aggiungere tqdm
                data_path = os.path.join(self.dataset_dir, data_files[i])
                if self.eurostat:
                    json_files = [ os.path.splitext(fname)[0] + '.json' for fname in data_files ]
                    json_path = os.path.join(self.dataset_dir, json_files[i])
                
                # Process the file with sample_rows parameter
                if self.eurostat:
                    output_file = process_csv_json_pair(self.client, data_path, json_path, output_dir)
                else:
                    output_file = process_csv_file(self.client, data_path, output_dir, entries_to_process)
                if output_file:
                    results.append(output_file)
            
            print(f"\nProcessed {len(results)} data files")
            print(f"Results saved to {output_dir}")
            
            """Function to generate a unified LinkML schema from all JSON files"""
            # Find the latest processor run
            latest_run_dir = output_dir
            
            if not latest_run_dir:
                print("No previous processor runs found. Please run generic_data_processor.py first.")
                return
            
            print(f"Processing JSON files from {latest_run_dir}")
        
            # Use the same output directory as the processor
            output_dir = latest_run_dir
            print(f"Output will be saved to the same directory: {output_dir}")
            
            # Get all JSON files from the latest run
            json_files = glob.glob(os.path.join(latest_run_dir, "*.json"))
            
            if not json_files:
                print("No JSON files found in the latest run directory.")
                return
            
            print(f"Found {len(json_files)} JSON files")
            
            # Step 1: Collect all column descriptions and examples from all JSON files
            all_columns = collect_all_column_data(json_files)
            print(f"Collected data for {len(all_columns)} unique columns")
            
            # Step 2: Unify descriptions for columns that have multiple descriptions
            unified_columns = unify_column_data(all_columns, self.client)
            print(f"Unified data for {len(unified_columns)} columns")
            
            # Step 3: Generate LinkML schema
            print("Generating LinkML schema...")
            schema_yaml = generate_linkml_schema(unified_columns, self.client)
            
            if not schema_yaml:
                print("Failed to generate LinkML schema")
                return
            
            # Step 4: Save the schema in the same directory as the processor output
            output_file = os.path.join(output_dir, "schema.yaml")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(schema_yaml)
            
            # Also save to yamls directory
            yamls_dir = os.path.join(self.output_folder, self.dataset_folder, self.method, "yamls")
            if not os.path.exists(yamls_dir):
                os.makedirs(yamls_dir, exist_ok = True)
            yamls_file = os.path.join(yamls_dir, "schema.yaml")
            with open(yamls_file, 'w', encoding='utf-8') as f:
                f.write(schema_yaml)
            
            self.schema_file = os.path.join(yamls_dir, "schema.yaml")
            
            print(f"LinkML schema saved to {output_file}")
            print(f"LinkML schema also saved to {yamls_file}")
        else:
            print(f"The embedding method '{self.method}' does not require any schema to be generaed.")
            return
    
    def pruneSchema(self, yaml_file = None) -> None:
        if self.method == "semantic_annotation":
            if yaml_file is None:
                if self.schema_file is None:
                    print(add_timestamp(color_text("No schema found", "red")))
                    return
                yaml_file = self.schema_file
            else:
                self.schema_file = yaml_file
            
            # Process all files in data directory
            print(add_timestamp(color_text("Pruning schema started..", "yellow")))
            
            yaml_classes_and_attributes, yaml_descriptions = yaml_parser(yaml_file)
            filtered_yaml_descriptions = {k: v for k, v in yaml_descriptions.items() if '.' not in k}
            
            # Verifica se c'è una sola classe nel YAML
            unique_classes = set()
            for class_name in filtered_yaml_descriptions.keys():
                # Estrai il nome della classe (prima del punto se presente)
                base_class = class_name.split('.')[0]
                unique_classes.add(base_class)
            
            # Se c'è una sola classe, assegna automaticamente quella classe a tutti i file
            if len(unique_classes) == 1:
                single_class = list(unique_classes)[0]
                print(add_timestamp(color_text(f"Only one class found in YAML: {single_class}. Defaulting all files to this class.", "yellow")))
                
                results = []
                for file_name in tqdm(os.listdir(self.dataset_dir)):
                    if file_name.endswith(('.csv', '.tsv')):
                        # Crea direttamente il risultato senza chiamare l'LLM
                        classification = {
                            "file_name": file_name,
                            "relevant_classes": [single_class],
                            "reasoning": ["Defaulted: only one class was present in the database"]
                        }
                        results.append(classification)
                        
                # Save results
                output_file = os.path.join(self.output_folder, self.dataset_folder, self.method, "pruning", f"classifications_{self.ts_run}.json")
                os.makedirs(os.path.dirname(output_file), exist_ok = True)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                self.pruned_schema_file = output_file
                print(f"\nResults saved to {output_file}")
                #return output_file
            
            # Se ci sono più classi, procedi con l'analisi normale
            results = []
            for file_name in tqdm(os.listdir(self.dataset_dir)):
                if file_name.endswith(('.csv', '.tsv')):
                    file_path = os.path.join(self.dataset_dir, file_name)
                    
                    # Process dataset
                    dataset_info = process_dataset(file_path)
                    if dataset_info:
                        # Get classification
                        classification = classify_columns(dataset_info, filtered_yaml_descriptions, self.client)
                        if classification:
                            results.append(classification)
            
            # Save results
            output_file = os.path.join(self.output_folder, self.dataset_folder, self.method, "pruning", f"classifications_{self.ts_run}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.pruned_schema_file = output_file
            print(f"\nResults saved to {output_file}")
            #return output_file
        else:
            print(f"The embedding method '{self.method}' does not require schema pruning.")
    
    def annotateDatasets(self, yaml_file = None, pruned_schema = None) -> None:
        
        if self.method == "semantic_annotation":
            if yaml_file is None:
                if self.schema_file is None:
                    print(add_timestamp(color_text("No schema found", "red")))
                    return
                yaml_file = self.schema_file
            else:
                self.schema_file = yaml_file
            if pruned_schema is None:
                if self.pruned_schema_file is None:
                    print(add_timestamp(color_text("No pruned schema found", "red")))
                    return
                pruned_schema = self.pruned_schema_file
            else:
                self.pruned_schema_file = pruned_schema
            
            output_dir = os.path.join(self.output_folder, self.dataset_folder, self.method, "annotations") 
            os.makedirs(output_dir, exist_ok=True)
            
            """ Run archetype on datasets using only relevant classes from previous classifications """
            print(add_timestamp(color_text("Running Archetype..", "yellow")))
            
            # Load previous classifications
            with open(self.pruned_schema_file, 'r') as f:
                classifications = json.load(f)
                
            print(f"Loaded {len(classifications)} classifications from {self.pruned_schema_file}")
            
            # Create lookup dict for relevant classes
            relevant_classes = {
                item['file_name']: item['relevant_classes'] 
                for item in classifications
            }
            
            print(f"Created lookup dictionary for {len(relevant_classes)} files")
            
            # Load YAML schema
            all_labels, _ = yaml_parser(yaml_file)
            
            print(f"Loaded YAML schema from {yaml_file} with {len(all_labels) if isinstance(all_labels, dict) else 'unknown'} labels")
        
            results = []
            
            # Process each file
            for filename in tqdm(os.listdir(self.dataset_dir)):
                if not (filename.endswith('.csv') or filename.endswith('.tsv')):
                    continue
                
                if filename not in relevant_classes:
                    print(f"No classification found for {filename}, skipping...")
                    continue
                
                print(f"\n{'='*50}")
                print(f"Processing {filename}...")
                
                # Filter custom_labels to only include relevant classes
                file_classes = relevant_classes[filename]
                print(f"Relevant classes for {filename}: {file_classes}")
                
                """ se non ci sono match tra i labels e le classi
                filtere labels è vuoto e archetype da errore """
                filtered_labels = {
                    k: v for k, v in all_labels.items() 
                    if k.split('.')[0] in file_classes
                } if isinstance(all_labels, dict) else [
                    label for label in all_labels 
                    if label.split('.')[0] in file_classes
                ]
                
                print(f"Filtered labels count: {len(filtered_labels) if isinstance(filtered_labels, dict) else len(filtered_labels)}")
                if isinstance(filtered_labels, dict):
                    print(f"Sample of filtered labels: {list(filtered_labels.keys())[:3]}")
                    
                # Read and sample dataset
                file_path = os.path.join(self.dataset_dir, filename)
                sep = '\t' if filename.endswith('.tsv') else ','
                
                print(f"Reading file: {file_path} with separator: '{sep}'")
                
                # Count rows and sample 
                total_rows = sum(1 for _ in open(file_path)) - 1
                n = 20  # sample size
                
                print(f"Total rows in file: {total_rows}, sampling {n} rows")

                df = pd.read_csv(file_path, sep = sep, dtype = str)
                print(f"Original columns: {df.columns.tolist()[:5]}{'...' if len(df.columns) > 5 else ''}")
                
                df_sample = df.sample(n = n,random_state=42)
                print(f"Sampled {len(df_sample)} rows")
                
                # Mask column names
                masked_columns = [f"Column_{i+1}" for i in range(len(df.columns))]
                df_sample.columns = masked_columns
                print(f"Masked columns: {masked_columns[:5]}{'...' if len(masked_columns) > 5 else ''}")
                
                if not filtered_labels:
                    filtered_labels = ["NAN"]
                
                # Setup Archetype with filtered labels
                args = {
                    "model_name": "gpt-4",
                    "custom_labels": filtered_labels,
                    "summ_stats": True,
                    "sample_size": 20
                }
                
                print(f"Setting up ArcheTypePredictor with model: {args['model_name']}")
                
                # Silent version of the OpenAI hook
                original_openai_chat_completion = self.client.chat.completions.create
                
                def silent_openai_chat_completion(*args, **kwargs):
                    # Executes the API call without printing anything
                    return original_openai_chat_completion(*args, **kwargs)
                
                # Temporarily replace the function
                openai.ChatCompletion.create = silent_openai_chat_completion
                
                # Silent version of the prompt generation hook
                from archetype.src.data import prompt_context_insert as original_prompt_context_insert
                
                def silent_prompt_context_insert(*args, **kwargs):
                    # Call the original function without printing anything
                    return original_prompt_context_insert(*args, **kwargs)
                
                # Temporarily replace the function
                import archetype.src.data
                archetype.src.data.prompt_context_insert = silent_prompt_context_insert
                
                print("Calling arch.annotate_columns() - This will invoke the LLM...")
                try:
                    arch = ArcheTypePredictor(input_files=[df_sample], user_args=args)
                    annotations = arch.annotate_columns()
                    print("LLM annotation complete!")
                finally:
                    # Restore original functions even in case of error
                    self.client.chat.completions.create = original_openai_chat_completion
                    archetype.src.data.prompt_context_insert = original_prompt_context_insert
                
                # Extract column annotations
                column_annotations = {
                    f"Column_{i+1}": annotations.columns[i] 
                    for i in range(len(annotations.columns))
                }
                
                print(f"Generated annotations for {len(column_annotations)} columns")
                print(f"Sample annotations: {list(column_annotations.items())[:3]}")
                
                # Read the original file
                file_path = os.path.join(self.dataset_dir, filename)
                sep = '\t' if filename.endswith('.tsv') else ','
                df = pd.read_csv(file_path, sep=sep, dtype=str)
                
                # Create new column names based on semantic annotations
                new_columns = []
                for i in range(len(df.columns)):
                    col_key = f"Column_{i+1}"
                    annotation = column_annotations.get(col_key, "")
                    # If annotation contains "context", set to NA
                    if "context" in annotation.lower():
                        new_columns.append("NA")
                    else:
                        new_columns.append(annotation)
                
                print(f"Initial semantic annotations: {new_columns[:5]}{'...' if len(new_columns) > 5 else ''}")
                
                # Track column name occurrences
                column_counts = {}
                for i, col in enumerate(new_columns):
                    if col in column_counts:
                        # Increment count and append to column name
                        column_counts[col] += 1
                        new_columns[i] = f"{col}.{column_counts[col]}"
                    else:
                        column_counts[col] = 1  # Start from 1 instead of 0
                
                print(f"Final semantic annotations (after deduplication): {new_columns[:5]}{'...' if len(new_columns) > 5 else ''}")
                
                results.append({
                    "file_name": filename,
                    "relevant_classes": file_classes,
                    "column_annotations": column_annotations
                })
                
                print(f"{'='*50}\n")
            
            # Save results JSON
            output_file_json = os.path.join(output_dir, f"annotations_{self.ts_run}.json")
            with open(output_file_json, 'w') as f:
                json.dump(results, f, indent=2)
            self.annotations_file = output_file_json
            
            print(f"\nResults saved to {output_file_json}")
        else:
            print(f"The embedding method '{self.method}' does not require any annotation to be done.")
            return
        
    def embedDatasets(self, yaml_file = None, annotations_file = None) -> list:
        if self.method == "semantic_annotation":
            if yaml_file is None:
                if self.schema_file is None:
                    print(add_timestamp(color_text("No schema found", "red")))
                    return
                yaml_file = self.schema_file
            else:
                self.schema_file = yaml_file
            if annotations_file is None:
                if self.annotations_file is None:
                    print(add_timestamp(color_text("No annotation file found", "red")))
                    return
                annotations_file = self.annotations_file
            else:
                self.annotations_file = annotations_file
        
        # ****** EMBEDDING ******
        SAFE_MAX_TOKENS=8000
        print(add_timestamp(color_text("Starting embeddings process", "blue")))
        print(add_timestamp(color_text(f"Using SAFE_MAX_TOKENS={SAFE_MAX_TOKENS}", "blue")))
        
        print(add_timestamp(color_text("Loading regular datasets", "blue")))
        datasets = [dataset for dataset in load_datasets(self.dataset_dir, self.method, annotations_file) 
                   if dataset['file_name'] not in self.file_blacklist]
        
        print(add_timestamp(color_text("Regenerating embeddings with OpenAI", "yellow")))

        total_columns = sum(len(d['columns']) for d in datasets)
        print(add_timestamp(color_text('Data directory: ' + self.dataset_dir, 'white')))
        
        print(add_timestamp(color_text(f"Total files: {len(datasets)}", "white")))
        print(add_timestamp(color_text(f"Total columns: {total_columns}", "white")))
        
        print('')
        print(add_timestamp(color_text("Blacklisted files: ", "red")))
        print(self.file_blacklist)
        print('')
        
        embeddings = []
        
        print(add_timestamp(color_text(f"Method: {self.method} \n", "cyan")))
        with tqdm(total=total_columns, desc=f"Getting embeddings for method: {self.method}", leave=False, position=0) as pbar:
            for data in datasets:
                file_name = data['file_name']
                # print(add_timestamp(color_text(f"Processing file: {file_name}", "blue")))
                
                try:
                    for column_dict in data['columns']:
                        
                        if column_dict["annotation"] == "NA" or column_dict["annotation"] == "nan" or "none" in column_dict["annotation"].lower() or "context" in column_dict["annotation"].lower():
                            continue
                        
                        if self.method == "semantic_annotation":
                            _, yaml_descriptions = yaml_parser(yaml_file)
                            text = prepare_text_with_semantic_annotation(column_dict, self.method, SAFE_MAX_TOKENS, yaml_descriptions)
                        else:
                            text = prepare_text_with_semantic_annotation(column_dict, self.method, SAFE_MAX_TOKENS)

                        if text is None:
                            print(add_timestamp(color_text("Text is None, skipping", "yellow")))
                            pbar.update(1)
                            continue

                        # Troncamento se necessario
                        num_tokens = count_tokens(text)
                        if num_tokens > SAFE_MAX_TOKENS:
                            text = truncate_text(text, SAFE_MAX_TOKENS)
                        
                        embedding = get_embedding(text, self.client)
                        if embedding is not None:
                            #print(add_timestamp(color_text("Successfully got embedding", "green")))
                            embeddings.append({
                                'column_name': f"{file_name}:{column_dict['column_name']}",
                                'embedding': embedding
                            })
                        else:
                            print(add_timestamp(color_text("Failed to get embedding", "red")))
                        

                        time.sleep(1)
                        pbar.update(1)
                except Exception as e:
                    print(add_timestamp(color_text(f"Error processing column {column_dict['column_name']} in file {file_name}: {e}", "red")))
                    print(e)
                finally:
                    pbar.update(1)
                    
        print(add_timestamp(color_text(f"Saving embeddings in pickle for method: {self.method} and ts_run: {self.ts_run}", "magenta")))
        # Save embeddings in a pickle file
        output_dir = os.path.join(self.output_folder, self.dataset_folder, self.method, "saved_embeddings", "pickle", self.ts_run, "openai")
        pickle_output_file = f"{output_dir}/embeddings.pkl"
        os.makedirs(output_dir, exist_ok = True)
        with open(pickle_output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(add_timestamp(color_text(f"Embeddings saved in: {pickle_output_file}", 'green')))
        self.embedding_file = pickle_output_file
        self.embeddings = embeddings
            
        return embeddings
    
    def computeDistances(self, embeddings = None) -> None:
        if embeddings is None:
            if self.embeddings is None:
                print(add_timestamp(color_text("No embeddings found", "red")))
                return
        
        """ Calculate cosine similarity and euclidean distance. """
        X = np.array([item['embedding'] for item in embeddings])

        # Calculate cosine similarity and euclidean distance
        cosine_sim_matrix = cosine_similarity(X)
        euclidean_dist_matrix = euclidean_distances(X)

        """ Get euclidean and cosine distances """
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
        sorted_distances = sorted(distances, key = lambda x: (x['euclidean_distance'], -x['cosine_similarity']))
    
        csv_output_dir = os.path.join(self.output_folder, self.dataset_folder, self.method, "saved_embeddings", "csv", "OTHER", "openai", self.ts_run)
        
        # Directory creation and save results
        os.makedirs(csv_output_dir, exist_ok=True)
        csv_output_file = os.path.join(csv_output_dir, "distances.csv")
        df_output = pd.DataFrame(sorted_distances)
        df_output.to_csv(csv_output_file, index = False, encoding = 'utf-8')
        print(add_timestamp(color_text(f"Euclid. and Cosine distances saved in: {csv_output_file}", 'green')))
        self.distances_file = csv_output_file
        self.distances = sorted_distances

        """ Get ANNS distances """
        # Calculates distances using ANNS and returns all combinations except self-comparisons.
        try:
            # Extract embeddings and column names
            X = np.array([item['embedding'] for item in embeddings], dtype = np.float32)  # Ensure float32
            column_names = [item['column_name'] for item in embeddings]

            # Create FAISS index with explicit thread control
            dimension = X.shape[1]
            faiss.omp_set_num_threads(1)  # Limit OpenMP threads
            index = faiss.IndexFlatL2(dimension)
            
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
            csv_output_dir = os.path.join(self.output_folder, self.dataset_folder, self.method, "saved_embeddings", "csv", "ANNS", "openai", self.ts_run)
            os.makedirs(csv_output_dir, exist_ok = True)
            csv_output_file = os.path.join(csv_output_dir, "anns_distances.csv")
            df_output = pd.DataFrame(sorted_results)
            df_output.to_csv(csv_output_file, index = False, encoding = 'utf-8')
            print(add_timestamp(color_text(f"ANNS distances saved in: {csv_output_file}", 'green')))

        finally:
            # Cleanup
            del index
            import gc
            gc.collect()
        
        self.anns_distances_file = csv_output_file
        self.anns_distances = sorted_results
    
    def saveGraph(self) -> None:
        # prepare for neo4j upload
        print(add_timestamp(color_text("Preparing for neo4j upload", "magenta")))
        print(add_timestamp(color_text("Preparing nodes", "cyan")))
        
        output_dir = os.path.join(self.output_folder, self.dataset_folder, self.method, "adalina", self.ts_run)
        os.makedirs(output_dir, exist_ok=True)
        
        nodes = prepare_nodes(self.embedding_file)
        nodes_file = os.path.join(output_dir, "nodes.csv")
        nodes.to_csv(nodes_file, index=False)
        print(add_timestamp(color_text(f"Nodes saved to {output_dir}/nodes.csv", "green")))
        self.nodes_file = nodes_file
        
        print(add_timestamp(color_text("Preparing edges", "cyan")))
        
        stats, relationships = prepare_edges(self.distances_file, self.anns_distances_file)
        
        mean_stas_dir = os.path.join(output_dir, "means_stats.json")
        with open(mean_stas_dir, "w") as f:
            json.dump(stats, f, indent=4)
        print(add_timestamp(color_text(f"Stats saved to {output_dir}/stats.json", "green")))
        
        relationships_file = os.path.join(output_dir, "edges.csv")
        relationships.to_csv(relationships_file, index=False)
        print(add_timestamp(color_text(f"Edges saved to {output_dir}/edges.csv", "green")))
        self.edges_file = relationships_file
        
        """ # Read the CSV file into a DataFrame
        df = pd.read_csv(relationships_file)
        mappings, _ = get_mappings(df)
        # Save the result to a JSON file
        mappings_file = os.path.join(self.output_folder, self.dataset_folder, self.method, "suggested_mappings.json")
        #df_out_file = os.path.join(self.output_folder, self.dataset_folder, self.method, "suggested_mappings.csv")
        #df.to_csv(df_out, index = False) # Export to CSV
        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=4)
        self.mappings_file = mappings_file
        self.mappings = mappings """