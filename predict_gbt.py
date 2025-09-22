import os
import json
import argparse
import numpy as np
import joblib
from gbt import load_cfgs, extract_node_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--java_file', required=True, help='Path to a Java slice to predict on')
    parser.add_argument('--model_path', required=True, help='Path to trained GBT model .joblib')
    parser.add_argument('--out_path', required=True, help='Path to write predictions JSON')
    parser.add_argument('--cfg_output_dir', default=os.environ.get('CFG_OUTPUT_DIR', 'cfg_output'))
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    cfgs = load_cfgs(args.java_file)
    results = []
    for cfg_data in cfgs:
        nodes = cfg_data['nodes']
        edges = cfg_data['edges']
        X = []
        idx_to_line = []
        for node in nodes:
            feats = extract_node_features(node, cfg_data, edges)
            if feats is None:
                X.append(None)
            else:
                X.append(feats)
            idx_to_line.append(node.get('line'))
        # Predict only for nodes with features
        valid_indices = [i for i, v in enumerate(X) if v is not None]
        if valid_indices:
            X_valid = np.array([X[i] for i in valid_indices])
            pred_valid = model.predict(X_valid)
            proba_valid = model.predict_proba(X_valid)[:, 1] if hasattr(model, 'predict_proba') else pred_valid
        else:
            pred_valid = np.array([])
            proba_valid = np.array([])
        # Rebuild full arrays
        pred = np.zeros(len(X), dtype=int)
        scores = np.zeros(len(X), dtype=float)
        for j, i in enumerate(valid_indices):
            pred[i] = int(pred_valid[j])
            scores[i] = float(proba_valid[j])
        results.append({
            'method': cfg_data.get('method_name'),
            'file': cfg_data.get('java_file'),
            'pred_lines': [int(idx_to_line[i]) for i in range(len(pred)) if idx_to_line[i] is not None and pred[i] == 1],
            'scores': [{ 'line': int(idx_to_line[i]) if idx_to_line[i] is not None else None, 'score': float(scores[i]) } for i in range(len(scores))]
        })

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Predictions written to {args.out_path}')

if __name__ == '__main__':
    main()


