#!/usr/bin/env python3
"""
GBT Prediction Script with Best Practices Defaults

This script runs GBT predictions on Java files or slices using:
- Dataflow-augmented CFGs by default
- Augmented slices when available
- Consistent pipeline integration

Best Practices:
- Uses dataflow information in CFGs for better predictions
- Prefers augmented slices over original slices
- Integrates with the same pipeline as training
"""

import os
import json
import argparse
import numpy as np
import joblib
from gbt import load_cfgs, extract_features_from_cfg

def main():
    parser = argparse.ArgumentParser(description='Run GBT predictions on Java files or slices with best practices defaults')
    parser.add_argument('--java_file', help='Path to a Java slice to predict on')
    parser.add_argument('--slices_dir', help='Path to directory containing Java slices (prefers augmented slices by default)')
    parser.add_argument('--model_path', required=True, help='Path to trained GBT model .joblib')
    parser.add_argument('--out_path', required=True, help='Path to write predictions JSON')
    parser.add_argument('--cfg_output_dir', default=os.environ.get('CFG_OUTPUT_DIR', 'cfg_output'),
                       help='Directory containing dataflow-augmented CFGs (default behavior)')
    parser.add_argument('--use_original_slices', action='store_true', default=False,
                       help='Use original slices instead of augmented slices (default: prefers augmented)')
    args = parser.parse_args()

    # Validate arguments
    if not args.java_file and not args.slices_dir:
        parser.error("Either --java_file or --slices_dir must be specified")
    if args.java_file and args.slices_dir:
        parser.error("Cannot specify both --java_file and --slices_dir")

    model = joblib.load(args.model_path)
    results = []
    
    # Process files
    if args.java_file:
        java_files = [args.java_file]
        print(f"Processing single file: {args.java_file}")
    else:
        java_files = []
        for root, dirs, files in os.walk(args.slices_dir):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        print(f"Processing {len(java_files)} files from {args.slices_dir}")
    
    for java_file in java_files:
        cfgs = load_cfgs(java_file, args.cfg_output_dir)
        for cfg_file in cfgs:
            cfg_data = cfg_file['data']
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


