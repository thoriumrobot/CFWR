import os
import json
import argparse
import numpy as np
import joblib
from causal_model import load_cfgs, extract_features_and_labels, parse_warnings, run_index_checker, preprocess_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--java_file', required=True, help='Path to a Java slice to predict on')
    parser.add_argument('--model_path', required=True, help='Path to trained causal classifier .joblib')
    parser.add_argument('--out_path', required=True, help='Path to write predictions JSON')
    args = parser.parse_args()

    clf = joblib.load(args.model_path)
    warnings_output = run_index_checker(args.java_file)
    annotations = parse_warnings(warnings_output)
    cfgs = load_cfgs(args.java_file)
    results = []
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


