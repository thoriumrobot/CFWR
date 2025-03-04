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
    command = [
        'javac',
        '-cp', index_checker_path,
        '-processor', 'org.checkerframework.checker.index.IndexChecker',
        java_file
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    warnings = result.stderr  # Warnings typically go to stderr
    return warnings

def parse_warnings(warnings):
    import re
    pattern = re.compile(r'^(.*\.java):(\d+):\s*(error|warning):\s*(.*)$')
    annotations = []
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
    X = []
    y = []

    # Build a set of annotation line numbers for quick lookup
    annotation_lines = set()
    # (Assumes all CFGs in cfgs are from the same file, or you adapt logic accordingly)
    if cfgs:
        original_java_file = os.path.abspath(cfgs[0]['java_file'])
        for ann in annotations:
            if os.path.abspath(ann['file']) == original_java_file:
                annotation_lines.add(ann['line'])

    for cfg_data in cfgs:
        nodes = cfg_data['nodes']
        edges = cfg_data['edges']
        # You might want total_lines for normalization
        total_lines = 1  # or compute from actual source lines

        for node in nodes:
            node_id = node['id']
            label = node['label']
            line_number = node.get('line', None)  # If we recorded line numbers

            # Create features
            features = extract_node_features(node, cfg_data, edges, total_lines)
            if features is None:
                continue

            # Label = 1 if line_number is in annotation_lines
            y_label = 1 if line_number in annotation_lines else 0

            X.append(features)
            y.append(y_label)

    return np.array(X), np.array(y)

def extract_node_features(node, cfg_data, edges, total_lines):
    label = node['label']
    line_number = node.get('line', None)
    if line_number is None:
        # If we don't have line info, skip
        return None

    normalized_line_pos = line_number / total_lines

    # Count in-degree and out-degree
    in_degree = 0
    out_degree = 0
    node_id = node['id']
    for edge in edges:
        if edge['target'] == node_id:
            in_degree += 1
        if edge['source'] == node_id:
            out_degree += 1

    # Example feature vector
    features = [
        len(label),       # label length
        in_degree,
        out_degree,
        normalized_line_pos
    ]
    return features

def train_model(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model

def save_model(model, iteration):
    model_path = os.path.join(models_dir, f"model_iteration_{iteration}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    best_accuracy = 0
    best_model = None

    all_X = []
    all_y = []

    # For each .java slice in slices_dir
    for java_file in os.listdir(slices_dir):
        if java_file.endswith(".java"):
            java_file_path = os.path.join(slices_dir, java_file)
            # Load CFGs for that slice
            cfgs = load_cfgs(java_file_path)
            if not cfgs:
                continue

            # Run Index Checker to get warnings
            warnings = run_index_checker(java_file_path)

            # Parse warnings => annotations
            annotations = parse_warnings(warnings)

            # Prepare data
            X, y = prepare_dataset(cfgs, annotations)
            if len(X) == 0:
                continue
            all_X.extend(X)
            all_y.extend(y)

    all_X = np.array(all_X)
    all_y = np.array(all_y)

    if len(all_X) == 0:
        print("No data found. Exiting...")
        exit(0)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        all_X, all_y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        save_model(model, 1)

    print(f"Best model accuracy: {best_accuracy}")
