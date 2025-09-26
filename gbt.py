import os
import subprocess
import json
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from cfg import generate_control_flow_graphs, save_cfgs

# Directory paths
java_project_dir = os.environ.get("JAVA_PROJECT_DIR", "")
index_checker_cp = os.environ.get("CHECKERFRAMEWORK_CP", "")
slices_dir = os.environ.get("SLICES_DIR", "slices")

# Default behavior: Use augmented slices if available, otherwise fall back to regular slices
def find_best_slices_directory():
    """Find the best available slices directory, preferring augmented slices."""
    # First, check if SLICES_DIR is explicitly set and contains Java files
    if os.path.exists(slices_dir) and any(f.endswith('.java') for f in os.listdir(slices_dir) if os.path.isfile(os.path.join(slices_dir, f))):
        return slices_dir
    
    # Look for augmented slices directories (preferred)
    base_dir = os.path.dirname(slices_dir) if os.path.dirname(slices_dir) else "."
    for slicer in ['specimin', 'wala']:  # Prefer specimin since it's working
        aug_dir = os.path.join(base_dir, f"slices_aug_{slicer}")
        if os.path.exists(aug_dir) and any(f.endswith('.java') for f in os.listdir(aug_dir) if os.path.isfile(os.path.join(aug_dir, f))):
            print(f"Using augmented slices from: {aug_dir}")
            return aug_dir
    
    # Look for general augmented slices directory
    aug_dir = os.path.join(base_dir, "slices_aug")
    if os.path.exists(aug_dir) and any(f.endswith('.java') for f in os.listdir(aug_dir) if os.path.isfile(os.path.join(aug_dir, f))):
        print(f"Using augmented slices from: {aug_dir}")
        return aug_dir
    
    # Fall back to regular slices
    if os.path.exists(slices_dir):
        print(f"Using regular slices from: {slices_dir}")
        return slices_dir
    
    # Last resort: look for any slices directory
    for potential_dir in ["slices", "slices_specimin", "slices_wala"]:
        if os.path.exists(potential_dir) and any(f.endswith('.java') for f in os.listdir(potential_dir) if os.path.isfile(os.path.join(potential_dir, f))):
            print(f"Using slices from: {potential_dir}")
            return potential_dir
    
    raise FileNotFoundError("No slices directory found with Java files")

slices_dir = find_best_slices_directory()
cfg_output_dir = os.environ.get("CFG_OUTPUT_DIR", "cfg_output")
models_dir = os.environ.get("MODELS_DIR", "models")

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
    command = ['javac']
    if index_checker_cp:
        command += ['-cp', index_checker_cp]
    command += ['-processor', 'org.checkerframework.checker.index.IndexChecker', java_file]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return result.stderr  # Warnings are typically in stderr
    except subprocess.TimeoutExpired:
        print(f"Timeout running Index Checker on {java_file}")
        return ""
    except Exception as e:
        print(f"Error running Index Checker on {java_file}: {e}")
        return ""

def parse_warnings(warnings_output):
    """
    Parse the warnings output from the Index Checker to extract annotation information.
    """
    annotations = []
    if not warnings_output:
        return annotations
    
    lines = warnings_output.split('\n')
    for line in lines:
        if 'warning:' in line.lower() and 'index' in line.lower():
            # Extract method name and annotation type from warning
            # This is a simplified parser - you may need to adjust based on actual warning format
            if 'method' in line.lower():
                parts = line.split()
                method_name = None
                annotation_type = None
                for i, part in enumerate(parts):
                    if 'method' in part.lower() and i + 1 < len(parts):
                        method_name = parts[i + 1].strip('(),')
                    if '@' in part:
                        annotation_type = part.strip('@')
                
                if method_name and annotation_type:
                    annotations.append({
                        'method': method_name,
                        'annotation': annotation_type,
                        'line': line
                    })
    
    return annotations

def extract_features_from_cfg(cfg_data):
    """
    Extract features from a CFG for machine learning.
    """
    try:
        # Basic graph features
        nodes = cfg_data.get('nodes', [])
        edges = cfg_data.get('edges', [])
        
        # Extract node labels
        node_labels = [node.get('label', '') for node in nodes if isinstance(node, dict)]
        
        features = [
            len(nodes),  # Number of nodes
            len(edges),  # Number of edges
            len([label for label in node_labels if 'if' in label.lower()]),  # Number of if statements
            len([label for label in node_labels if 'for' in label.lower()]),  # Number of for loops
            len([label for label in node_labels if 'while' in label.lower()]),  # Number of while loops
            len([label for label in node_labels if 'try' in label.lower()]),  # Number of try blocks
            len([label for label in node_labels if 'switch' in label.lower()]),  # Number of switch statements
            len([label for label in node_labels if 'return' in label.lower()]),  # Number of return statements
        ]
        
        return features
    except Exception as e:
        print(f"Error extracting features from CFG: {e}")
        return None

def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Classifier.
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, model_id):
    """
    Save the trained model to disk.
    """
    model_path = os.path.join(models_dir, f'gbt_model_{model_id}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return accuracy.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def iter_java_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.java'):
                yield os.path.join(root, f)

def main():
    # Collect data from all slices
    all_X = []
    all_y = []

    for java_file_path in iter_java_files(slices_dir):
        # Ensure CFGs exist
        base = os.path.splitext(os.path.basename(java_file_path))[0]
        out_dir = os.path.join(cfg_output_dir, base)
        if not os.path.exists(out_dir) or not any(name.endswith('.json') for name in os.listdir(out_dir)):
            cfgs_gen = generate_control_flow_graphs(java_file_path, cfg_output_dir)
            save_cfgs(cfgs_gen, out_dir)
        cfgs = load_cfgs(java_file_path)
        if not cfgs:
            continue
        # Extract features from CFGs and generate synthetic labels
        for cfg_data in cfgs:
            features = extract_features_from_cfg(cfg_data)
            if features is not None:
                all_X.append(features)
                # Generate synthetic labels based on CFG complexity
                # Methods with more complex control flow are more likely to need annotations
                complexity_score = sum(features[2:])  # Sum of control flow features
                needs_annotation = 1 if complexity_score > 2 else 0
                all_y.append(needs_annotation)

    if len(set(all_y)) < 2:
        print("GBT: Not enough class diversity to train (need >=2 classes). Skipping GBT training.")
        return

    # Convert to numpy arrays
    X = np.array(all_X)
    y = np.array(all_y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    # Save the model
    save_model(model, 1)
    print(f"GBT training completed with accuracy: {accuracy}")

if __name__ == "__main__":
    main()