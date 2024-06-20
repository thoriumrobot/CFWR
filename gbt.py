# gbt.py
import os
import subprocess
import networkx as nx
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from cfg import generate_control_flow_graphs

java_project_dir = "path/to/java/project"
index_checker_path = "path/to/checker-framework/index-checker.jar"
slices_dir = "path/to/slices"
models_dir = "path/to/models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def run_index_checker(java_file):
    result = subprocess.run(['java', '-jar', index_checker_path, java_file], capture_output=True, text=True)
    warnings = result.stdout
    return warnings

def parse_warnings(warnings):
    annotations = []
    for warning in warnings.split('\n'):
        parts = warning.split(':')
        if len(parts) > 3:
            file = parts[0]
            line = int(parts[1])
            column = int(parts[2])
            annotations.append((file, line, column))
    return annotations

def prepare_dataset(cfgs, annotations):
    X = []
    y = []
    for cfg, annotation in zip(cfgs, annotations):
        features = extract_features(cfg, annotation)
        label = determine_label(cfg, annotation)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def extract_features(cfg, annotation):
    node_features = []
    for node in cfg.nodes(data=True):
        feature = [cfg.degree(node[0])]
        feature.append(node[1].get('label', 'unknown'))
        node_features.append(feature)
    return np.array(node_features).flatten()

def determine_label(cfg, annotation):
    for node in cfg.nodes(data=True):
        if node[1].get('label') == annotation:
            return 1
    return 0

def train_model(X, y):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    return model

def save_model(model, iteration):
    model_path = os.path.join(models_dir, f"model_iteration_{iteration}.joblib")
    joblib.dump(model, model_path)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

best_accuracy = 0
best_model = None

for iteration in range(10):
    cfgs = []
    annotations = []

    for java_file in os.listdir(slices_dir):
        if java_file.endswith(".java"):
            cfg = generate_control_flow_graphs(os.path.join(slices_dir, java_file))
            warnings = run_index_checker(os.path.join(slices_dir, java_file))
            annotation = parse_warnings(warnings)
            cfgs.append(cfg)
            annotations.append(annotation)

    X, y = prepare_dataset(cfgs, annotations)
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        save_model(model, iteration)

print(f"Best model accuracy: {best_accuracy}")
