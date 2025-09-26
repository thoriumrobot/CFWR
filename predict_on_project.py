#!/usr/bin/env python3
"""
Enhanced prediction pipeline that can work on warning-based slices of target projects.
This script integrates slicing, CFG generation, and prediction for comprehensive analysis.
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
import joblib
from torch_geometric.data import DataLoader

# Import our modules
from hgt import HGTModel, create_heterodata, load_cfgs
from gbt import load_cfgs as load_cfgs_gbt, extract_features_from_cfg
from causal_model import load_cfgs as load_cfgs_causal, extract_features_and_labels, parse_warnings, run_index_checker, preprocess_data
from cfg import generate_control_flow_graphs, save_cfgs

def run_checker_framework_on_project(project_root, output_file):
    """Run Checker Framework on a project and save warnings to output file."""
    print(f"Running Checker Framework on project: {project_root}")
    
    # Set up environment
    env = os.environ.copy()
    checker_home = env.get('CHECKERFRAMEWORK_HOME', '/home/ubuntu/checker-framework-3.42.0')
    checker_cp = env.get('CHECKERFRAMEWORK_CP', '')
    
    # Build the javac command with Checker Framework
    cmd = [
        'javac',
        '-cp', checker_cp,
        '-processor', 'org.checkerframework.checker.index.IndexChecker',
        '-Xmaxwarns', '1000',  # Limit warnings to avoid overwhelming output
        '-d', '/tmp/checker_output',
        '-sourcepath', project_root
    ]
    
    # Find all Java files in the project
    java_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip test directories and build directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['test', 'tests', 'target', 'build']]
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    if not java_files:
        print(f"No Java files found in {project_root}")
        return False
    
    cmd.extend(java_files[:10])  # Limit to first 10 files for testing
    
    try:
        # Create output directory
        os.makedirs('/tmp/checker_output', exist_ok=True)
        
        # Run the checker
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        # Save warnings to output file
        with open(output_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        
        print(f"Checker Framework completed. Warnings saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error running Checker Framework: {e}")
        return False

def generate_slices_from_warnings(warnings_file, project_root, output_dir, slicer_type='specimin'):
    """Generate slices from Checker Framework warnings."""
    print(f"Generating slices from warnings using {slicer_type} slicer...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up environment for the resolver
    env = os.environ.copy()
    env['SLICES_DIR'] = output_dir
    
    # Run the CheckerFrameworkWarningResolver
    cmd = [
        'java', '-cp', 
        f'{os.getcwd()}/build/libs/CFWR-1.0-SNAPSHOT.jar:{env.get("CHECKERFRAMEWORK_CP", "")}',
        'cfwr.CheckerFrameworkWarningResolver',
        project_root,
        warnings_file,
        output_dir,
        slicer_type
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"Warning: Slicer returned non-zero exit code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
        
        print(f"Slicing completed. Slices saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error running slicer: {e}")
        return False

def generate_cfgs_for_slices(slices_dir, cfg_output_dir):
    """Generate CFGs for all slices in the directory."""
    print(f"Generating CFGs for slices in {slices_dir}")
    
    os.makedirs(cfg_output_dir, exist_ok=True)
    
    java_files = []
    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    print(f"Found {len(java_files)} Java files to process")
    
    for java_file in java_files:
        try:
            base_name = os.path.splitext(os.path.basename(java_file))[0]
            cfg_dir = os.path.join(cfg_output_dir, base_name)
            
            if not os.path.exists(cfg_dir) or not any(f.endswith('.json') for f in os.listdir(cfg_dir)):
                print(f"Generating CFGs for {java_file}")
                cfgs = generate_control_flow_graphs(java_file, cfg_output_dir)
                save_cfgs(cfgs, cfg_dir)
        except Exception as e:
            print(f"Error processing {java_file}: {e}")
    
    print(f"CFG generation completed. CFGs saved to {cfg_output_dir}")

def run_hgt_predictions(slices_dir, cfg_output_dir, model_path, output_file):
    """Run HGT predictions on slices."""
    print("Running HGT predictions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model - need to reconstruct the model architecture first
    model_state = torch.load(model_path, map_location=device)
    
    # We need to create a sample graph first to get model metadata
    sample_graphs = []
    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.java'):
                java_file = os.path.join(root, file)
                try:
                    cfgs = load_cfgs(java_file)
                    for cfg_data in cfgs:
                        data = create_heterodata(cfg_data)
                        if data is not None:
                            sample_graphs.append((cfg_data, data))
                            break
                    if sample_graphs:
                        break
                except Exception as e:
                    continue
        if sample_graphs:
            break
    
    if not sample_graphs:
        print("No valid graphs found for HGT prediction")
        return
    
    # Build model with correct architecture
    sample_graph = sample_graphs[0][1]
    metadata = sample_graph.metadata()
    in_channels = sample_graph['node'].x.size(-1)
    model = HGTModel(in_channels=in_channels, hidden_channels=64, out_channels=2, num_heads=2, num_layers=2, metadata=metadata)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    results = []
    
    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.java'):
                java_file = os.path.join(root, file)
                try:
                    cfgs = load_cfgs(java_file)
                    for cfg_data in cfgs:
                        data = create_heterodata(cfg_data)
                        if data is not None:
                            data = data.to(device)
                            with torch.no_grad():
                                out = model(data.x_dict, data.edge_index_dict)
                                prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                                pred = out.argmax(dim=1).cpu().numpy()
                                
                                lines = [node.get('line') for node in cfg_data['nodes']]
                                results.append({
                                    'method': cfg_data.get('method_name'),
                                    'file': cfg_data.get('java_file'),
                                    'pred_lines': [int(lines[i]) for i in range(len(pred)) if lines[i] is not None and pred[i] == 1],
                                    'scores': [{'line': int(lines[i]) if lines[i] is not None else None, 'score': float(prob[i])} for i in range(len(lines))]
                                })
                except Exception as e:
                    print(f"Error processing {java_file} for HGT: {e}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"HGT predictions written to {output_file}")

def run_gbt_predictions(slices_dir, cfg_output_dir, model_path, output_file):
    """Run GBT predictions on slices."""
    print("Running GBT predictions...")
    
    model = joblib.load(model_path)
    results = []
    
    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.java'):
                java_file = os.path.join(root, file)
                try:
                    cfgs = load_cfgs_gbt(java_file)
                    for cfg_data in cfgs:
                        feats = extract_features_from_cfg(cfg_data)
                        if feats is not None:
                            X = np.array([feats])
                            pred = model.predict(X)[0]
                            proba = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else pred
                            
                            nodes = cfg_data.get('nodes', [])
                            results.append({
                                'method': cfg_data.get('method_name'),
                                'file': cfg_data.get('java_file'),
                                'pred_lines': [int(node.get('line')) for node in nodes if node.get('line') is not None and pred == 1],
                                'scores': [{'line': int(node.get('line')) if node.get('line') is not None else None, 'score': float(proba)} for node in nodes]
                            })
                except Exception as e:
                    print(f"Error processing {java_file} for GBT: {e}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"GBT predictions written to {output_file}")

def run_causal_predictions(slices_dir, cfg_output_dir, model_path, output_file):
    """Run Causal model predictions on slices."""
    print("Running Causal model predictions...")
    
    clf = joblib.load(model_path)
    results = []
    
    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.java'):
                java_file = os.path.join(root, file)
                try:
                    warnings_output = run_index_checker(java_file)
                    annotations = parse_warnings(warnings_output)
                    cfgs = load_cfgs_causal(java_file)
                    
                    for cfg_data in cfgs:
                        records = extract_features_and_labels(cfg_data, annotations)
                        if records:
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
                except Exception as e:
                    print(f"Error processing {java_file} for Causal: {e}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Causal predictions written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run predictions on warning-based slices of a target project')
    parser.add_argument('--project_root', required=True, help='Root directory of the target project')
    parser.add_argument('--models_dir', default='models', help='Directory containing trained models')
    parser.add_argument('--output_dir', default='predictions_project', help='Output directory for predictions')
    parser.add_argument('--slicer', choices=['specimin', 'wala'], default='specimin', help='Slicer to use')
    parser.add_argument('--models', nargs='+', choices=['hgt', 'gbt', 'causal'], default=['hgt', 'gbt', 'causal'], help='Models to run predictions with')
    parser.add_argument('--skip_checker', action='store_true', help='Skip Checker Framework analysis (use existing warnings)')
    parser.add_argument('--warnings_file', help='Path to existing warnings file (if skipping checker)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Run Checker Framework (unless skipped)
    warnings_file = args.warnings_file or os.path.join(args.output_dir, 'warnings.out')
    if not args.skip_checker:
        if not run_checker_framework_on_project(args.project_root, warnings_file):
            print("Failed to run Checker Framework. Exiting.")
            return 1
    
    # Step 2: Generate slices from warnings
    slices_dir = os.path.join(args.output_dir, f'slices_{args.slicer}')
    if not generate_slices_from_warnings(warnings_file, args.project_root, slices_dir, args.slicer):
        print("Failed to generate slices. Exiting.")
        return 1
    
    # Step 3: Generate CFGs for slices
    cfg_output_dir = os.path.join(args.output_dir, 'cfgs')
    generate_cfgs_for_slices(slices_dir, cfg_output_dir)
    
    # Step 4: Run predictions with specified models
    for model_type in args.models:
        model_path = os.path.join(args.models_dir, {
            'hgt': 'best_model.pth',
            'gbt': 'gbt_model_1.joblib',
            'causal': 'causal_clf.joblib'
        }[model_type])
        
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Skipping {model_type} predictions.")
            continue
        
        output_file = os.path.join(args.output_dir, f'predictions_{model_type}.json')
        
        if model_type == 'hgt':
            run_hgt_predictions(slices_dir, cfg_output_dir, model_path, output_file)
        elif model_type == 'gbt':
            run_gbt_predictions(slices_dir, cfg_output_dir, model_path, output_file)
        elif model_type == 'causal':
            run_causal_predictions(slices_dir, cfg_output_dir, model_path, output_file)
    
    print(f"\nPrediction pipeline completed. Results saved to {args.output_dir}")
    return 0

if __name__ == '__main__':
    exit(main())
