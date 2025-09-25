import os
import json
import argparse
import numpy as np
import joblib
from gbt import load_cfgs, extract_features_from_cfg

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
        # Extract features for the entire CFG
        feats = extract_features_from_cfg(cfg_data)
        if feats is None:
            continue
            
        # Make prediction for the CFG
        X = np.array([feats])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else pred
        
        # For GBT, we predict at the CFG level, so we'll assign the same score to all nodes
        nodes = cfg_data.get('nodes', [])
        results.append({
            'method': cfg_data.get('method_name'),
            'file': cfg_data.get('java_file'),
            'pred_lines': [int(node.get('line')) for node in nodes if node.get('line') is not None and pred == 1],
            'scores': [{ 'line': int(node.get('line')) if node.get('line') is not None else None, 'score': float(proba) } for node in nodes]
        })

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Predictions written to {args.out_path}')

if __name__ == '__main__':
    main()


