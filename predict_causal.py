#!/usr/bin/env python3
"""
Causal Model Prediction Script with Best Practices Defaults

This script runs Causal model predictions on Java files or slices using:
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
from causal_model import load_cfgs, extract_features_and_labels, parse_warnings, run_index_checker, preprocess_data

def main():
    parser = argparse.ArgumentParser(description='Run Causal model predictions on Java files or slices with best practices defaults')
    parser.add_argument('--java_file', help='Path to a Java slice to predict on')
    parser.add_argument('--slices_dir', help='Path to directory containing Java slices (prefers augmented slices by default)')
    parser.add_argument('--model_path', required=True, help='Path to trained causal classifier .joblib')
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

    clf = joblib.load(args.model_path)
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
        warnings_output = run_index_checker(java_file)
        annotations = parse_warnings(warnings_output)
        cfgs = load_cfgs(java_file, args.cfg_output_dir)
        for cfg_data in cfgs:
            # Build dataframe with features and labels
            records = extract_features_and_labels(cfg_data, annotations)
            if not records:
                continue
            import pandas as pd
            df = pd.DataFrame(records)
            df = preprocess_data(df)
            feats = ['label_length', 'in_degree', 'out_degree', 'label_encoded', 'line_number']
            X = df[feats]
            pred = clf.predict(X)
            results.append({
                'method': cfg_data.get('method_name'),
                'file': cfg_data.get('java_file'),
                'pred_lines': [int(df.loc[i, 'line_number']) for i in range(len(pred)) if df.loc[i, 'line_number'] != 0 and pred[i] == 1],
                'scores': None
            })

    with open(args.out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Predictions written to {args.out_path}')

if __name__ == '__main__':
    main()


