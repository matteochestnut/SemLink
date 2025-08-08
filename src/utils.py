import json
import time
import os
import yaml
from colorama import Fore, Style

# Suppress KMP_DUPLICATE_LIB_OK warning and set OMP_NUM_THREADS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

def color_text(text: str, color: str = 'white') -> str:
    """
    Colors text using colorama colors.

    Args:
        text (str): Text to be colored.
        color (str): Color name to use, defaults to 'white'.
                     Valid colors: 'black', 'red', 'green', 'yellow', 'blue',
                     'magenta', 'cyan', 'white'.

    Returns:
        str: Colored text string with reset style appended.
    """
    colors = {
        'black': Fore.BLACK,
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE
    }
    color_code = colors.get(color.lower(), Fore.WHITE)
    return f"{color_code}{text}{Style.RESET_ALL}"

def add_timestamp(text: str) -> str:
    """
    Adds a timestamp before the provided text.

    Args:
        text (str): Text to prepend timestamp to.

    Returns:
        str: Text with timestamp prepended in format [HH:MM:SS].
    """
    timestamp = time.strftime("[%H:%M:%S]")
    return f"{timestamp} {text}"

def create_directory_if_not_exists(path: str):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): The path to the directory to create.
    """
    os.makedirs(path, exist_ok=True)
    print(add_timestamp(color_text(f"Ensured directory exists: {path}", "green")))

def load_datalake_json(json_file_path: str) -> list[dict]:
    """
    Loads a standardized data lake list of dictionaries from a JSON file.

    Args:
        json_file_path (str): The full path to the data lake JSON file.

    Returns:
        list[dict]: The loaded list of dictionaries representing the data lake.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_lake_list = json.load(f)
        print(add_timestamp(color_text(f"Data lake JSON loaded from: {json_file_path}", "green")))
        return data_lake_list
    except FileNotFoundError:
        print(add_timestamp(color_text(f"Error: Data lake JSON file not found at {json_file_path}", "red")))
        return []
    except json.JSONDecodeError as e:
        print(add_timestamp(color_text(f"Error decoding JSON from {json_file_path}: {e}", "red")))
        return []
    except Exception as e:
        print(add_timestamp(color_text(f"Error loading data lake JSON from {json_file_path}: {e}", "red")))
        return []

def yaml_parser(data: dict) -> tuple[list[str], dict[str, str]]:
    """
    Parses a YAML schema file to extract classes, attributes, and their descriptions.
    It handles inheritance for attributes with special handling for root classes.

    Args:
        data (dict): The loaded YAML schema data.

    Returns:
        tuple[list[str], dict[str, str]]:
            - A list of strings in the format "ClassName.attributeName" for all
              classes and their attributes, including inherited ones.
            - A dictionary mapping "ClassName" or "ClassName.attributeName" to their descriptions.
    """
    classes_and_attributes_labels = []  # list of class.attribute for every class and its attributes
    descriptions = {}

    all_classes = data.get('classes', {})
    
    # Identify root classes (those inheriting directly from NamedEntity or Triple)
    root_classes = {
        class_name: details
        for class_name, details in all_classes.items()
        if details.get('is_a') in ["NamedEntity", "Triple"]
    }
    
    # Identify hereditary classes (all others)
    hereditary_classes = {
        class_name: details
        for class_name, details in all_classes.items()
        if class_name not in root_classes
    }

    # First pass: Collect all class descriptions
    for class_name, details in all_classes.items():
        if "description" in details and details["description"]:
            descriptions[class_name] = details["description"]

    # Process root classes first
    for class_name, details in root_classes.items():
        current_class_attributes = []
        
        if "attributes" in details:
            for attr_name, attr_details in details["attributes"].items():
                # Skip specific attributes that are part of LinkML's internal structure
                if attr_name in ["subject", "object", "predicate"]:
                    continue
                current_class_attributes.append(attr_name)
                if "description" in attr_details and attr_details["description"]:
                    descriptions[f"{class_name}.{attr_name}"] = attr_details["description"]

        # Add all collected attributes for the current class to the main list
        for attribute in current_class_attributes:
            classes_and_attributes_labels.append(f"{class_name}.{attribute}")

    # Process hereditary classes with inheritance handling
    for class_name, details in hereditary_classes.items():
        current_class_attributes = []
        inherited_attributes = []
        
        # Add attributes directly defined in the current class
        if "attributes" in details:
            for attr_name, attr_details in details["attributes"].items():
                if attr_name in ["subject", "object", "predicate"]:
                    continue
                current_class_attributes.append(attr_name)
                inherited_attributes.append(attr_name)
                if "description" in attr_details and attr_details["description"]:
                    descriptions[f"{class_name}.{attr_name}"] = attr_details["description"]

        # Handle inheritance chain
        is_a = details.get('is_a')
        while is_a and is_a in all_classes:
            parent_class = all_classes[is_a]
            
            # Add parent's attributes
            if "attributes" in parent_class:
                for attr_name, attr_details in parent_class["attributes"].items():
                    if attr_name in ["subject", "object", "predicate"]:
                        continue
                    if attr_name not in inherited_attributes:  # Avoid duplicates
                        inherited_attributes.append(attr_name)
                        if "description" in attr_details and attr_details["description"]:
                            descriptions[f"{class_name}.{attr_name}"] = attr_details["description"]
            
            # Move up the inheritance chain
            is_a = parent_class.get('is_a')
            if is_a in ["NamedEntity", "Triple"]:  # Stop at root classes
                break

        # Add all collected attributes for the current class to the main list
        for attribute in inherited_attributes:
            classes_and_attributes_labels.append(f"{class_name}.{attribute}")

    print(add_timestamp(color_text(
        f"Loaded YAML file with {len(classes_and_attributes_labels)} class.attribute pairs "
        f"and {len(descriptions)} descriptions.", "green"
    )))
    return classes_and_attributes_labels, descriptions