import os
import subprocess
import json
import networkx as nx
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from cfg import generate_control_flow_graphs

# Directory paths
java_project_dir = "path/to/java/project"  # Replace with the actual path
index_checker_path = "path/to/checker-framework/index.jar"  # Replace with the actual path
slices_dir = "path/to/slices"  # Directory containing Java code slices
cfg_output_dir = "cfg_output"  # Directory where CFGs are saved
models_dir = "models"  # Directory to save models

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def load_cfgs(java_file):
    """
    Load the saved CFGs for a given Java file.
    """
    method_cfgs = []
    java_file_name = os.path.splitext(os.path.basename(java_file))[0]
    cfg_dir = os.path.join(cfg_output_dir, java_file_name)
    if os.path.exists(cfg_dir):
        for cfg_file in os.listdir(cfg_dir):
            if cfg_file.endswith('.json'):
                cfg_file_path = os.path.join(cfg_dir, cfg_file)
                with open(cfg_file_path, 'r') as f:
                    cfg_data = json.load(f)
                    # Add method name to cfg_data for identification
                    cfg_data['method_name'] = os.path.splitext(cfg_file)[0]
                    cfg_data['java_file'] = java_file
                    method_cfgs.append(cfg_data)
    else:
        print(f"CFG directory {cfg_dir} does not exist for Java file {java_file}")
    return method_cfgs

def run_index_checker(java_file):
    """
    Run the Checker Framework's Index Checker on the given Java file and capture warnings.
    """
    # Construct the command to run the Index Checker
    command = [
        'javac',
        '-cp', index_checker_path,
        '-processor', 'org.checkerframework.checker.index.IndexChecker',
        java_file
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    warnings = result.stderr  # Warnings are typically output to stderr
    return warnings

def parse_warnings(warnings):
    """
    Parse the warnings generated by Index Checker to identify nodes for annotations.
    """
    import re
    pattern = re.compile(r'^(.*\.java):(\d+):\s*(error|warning):\s*(.*)$')
    annotations = []  # List of dictionaries with file, line, and message

    for line in warnings.split('\n'):
        match = pattern.match(line)
        if match:
            file_path = match.group(1).strip()
            line_number = int(match.group(2).strip())
            message_type = match.group(3).strip()
            message = match.group(4).strip()
            annotations.append({
                'file': file_path,
                'line': line_number,
                'message_type': message_type,
                'message': message
            })
    return annotations

def prepare_dataset(cfgs, annotations):
    """
    Prepare the dataset from control flow graphs and annotations for training the model.
    """
    X = []
    y = []

    # Build a set of annotation line numbers for quick lookup
    annotation_lines = set()
    for annotation in annotations:
        if os.path.abspath(annotation['file']) == os.path.abspath(cfgs[0]['java_file']):
            annotation_lines.add(annotation['line'])

    for cfg_data in cfgs:
        nodes = cfg_data['nodes']
        edges = cfg_data['edges']
        for node in nodes:
            node_id = node['id']
            label = node['label']
            line_number = node.get('line', None)

            # Extract features from the node
            features = extract_node_features(node, cfg_data, edges)
            if features is None:
                continue  # Skip nodes without valid features

            # Label the node: 1 if an annotation is needed, 0 otherwise
            if line_number in annotation_lines:
                y_label = 1
            else:
                y_label = 0

            X.append(features)
            y.append(y_label)
    return np.array(X), np.array(y)

def extract_node_features(node, cfg_data, edges):
    """
    Extract features from a CFG node for training.
    """
    # Features can include:
    # - Node label length
    # - Node degree (in-degree and out-degree)
    # - Whether the node represents a specific statement type
    # - Position in the method (normalized line number)

    label = node['label']
    line_number = node.get('line', None)

    if line_number is None:
        return None  # Skip nodes without line numbers

    total_lines = cfg_data.get('total_lines', 1)
    normalized_line_pos = line_number / total_lines

    in_degree = 0
    out_degree = 0
    node_id = node['id']
    for edge in edges:
        if edge['target'] == node_id:
            in_degree += 1
        if edge['source'] == node_id:
            out_degree += 1

    # Simple feature vector
    features = [
        len(label),
        in_degree,
        out_degree,
        normalized_line_pos
    ]

    return features

def train_model(X_train, y_train):
    """
    Train a Gradient Boosted Trees model.
    """
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model

def save_model(model, iteration):
    """
    Save the trained model to a file.
    """
    model_path = os.path.join(models_dir, f"model_iteration_{iteration}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return accuracy.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Main training loop
best_accuracy = 0
best_model = None

# Collect data from all slices
all_X = []
all_y = []

for java_file in os.listdir(slices_dir):
    if java_file.endswith(".java"):
        java_file_path = os.path.join(slices_dir, java_file)
        # Load CFGs
        cfgs = load_cfgs(java_file_path)
        if not cfgs:
            continue
        # Run Index Checker
        warnings = run_index_checker(java_file_path)
        # Parse warnings
        annotations = parse_warnings(warnings)
        # Prepare dataset
        X, y = prepare_dataset(cfgs, annotations)
        all_X.extend(X)
        all_y.extend(y)

# Convert to numpy arrays
all_X = np.array(all_X)
all_y = np.array(all_y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)
accuracy = evaluate_model(model, X_test, y_test)
print(f"Model accuracy: {accuracy}")

# Save the model if it's better than the previous best
if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model = model
    save_model(model, 1)

print(f"Best model accuracy: {best_accuracy}")
