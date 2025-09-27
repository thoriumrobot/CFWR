#!/usr/bin/env python3
"""
Prediction Script with Checker Framework CFG Builder

This script runs predictions using Checker Framework CFG Builder for slicing,
following the user's requirement to use CFG Builder for prediction.
"""

import os
import subprocess
import sys
import argparse
import json
from pathlib import Path

def setup_cfg_builder_environment():
    """Set up environment variables for CFG Builder prediction"""
    # Set environment variables for CFG Builder-based prediction
    os.environ['SLICES_DIR'] = '/home/ubuntu/CFWR/slices_cf'
    os.environ['AUGMENTED_SLICES_DIR'] = '/home/ubuntu/CFWR/slices_aug_cf'
    os.environ['CFG_OUTPUT_DIR'] = '/home/ubuntu/CFWR/cfg_output_cf'
    os.environ['MODELS_DIR'] = '/home/ubuntu/CFWR/models_specimin'  # Use models trained with Specimin
    
    print("Environment configured for CFG Builder prediction:")
    print(f"  SLICES_DIR: {os.environ['SLICES_DIR']}")
    print(f"  AUGMENTED_SLICES_DIR: {os.environ['AUGMENTED_SLICES_DIR']}")
    print(f"  CFG_OUTPUT_DIR: {os.environ['CFG_OUTPUT_DIR']}")
    print(f"  MODELS_DIR: {os.environ['MODELS_DIR']} (using Specimin-trained models)")

def run_cfg_builder_slicing(project_root, warnings_file):
    """Run slicing using Checker Framework CFG Builder"""
    print("Step 1: Running CFG Builder slicing...")
    
    # Create directories
    os.makedirs(os.environ['SLICES_DIR'], exist_ok=True)
    os.makedirs(os.environ['AUGMENTED_SLICES_DIR'], exist_ok=True)
    os.makedirs(os.environ['CFG_OUTPUT_DIR'], exist_ok=True)
    
    # Run pipeline with CF slicer (CFG Builder)
    cmd = [
        sys.executable, 'pipeline.py',
        '--steps', 'slice,augment,cfg',
        '--project_root', project_root,
        '--warnings_file', warnings_file,
        '--slicer', 'cf',  # Use Checker Framework CFG Builder
        '--output_dir', '/home/ubuntu/CFWR/prediction_output_cf'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"CFG Builder slicing failed: {result.stderr}")
        return False
    
    print("CFG Builder slicing completed successfully")
    return True

def run_predictions(models):
    """Run predictions using CFG Builder-generated slices"""
    print("Step 2: Running predictions with CFG Builder slices...")
    
    predictions_dir = '/home/ubuntu/CFWR/predictions_cf'
    os.makedirs(predictions_dir, exist_ok=True)
    
    prediction_files = {}
    
    for model in models:
        print(f"Running {model.upper()} predictions...")
        
        # Determine model file path
        models_dir = os.environ['MODELS_DIR']
        if model == 'hgt':
            model_path = os.path.join(models_dir, 'best_model.pth')
            output_file = os.path.join(predictions_dir, 'hgt_predictions_cf.json')
            cmd = [sys.executable, 'predict_hgt.py', 
                   '--slices_dir', os.environ['SLICES_DIR'],
                   '--model_path', model_path,
                   '--out_path', output_file]
        elif model == 'gbt':
            model_path = os.path.join(models_dir, 'model_iteration_1.joblib')
            output_file = os.path.join(predictions_dir, 'gbt_predictions_cf.json')
            cmd = [sys.executable, 'predict_gbt.py',
                   '--slices_dir', os.environ['SLICES_DIR'],
                   '--model_path', model_path,
                   '--out_path', output_file]
        elif model == 'causal':
            model_path = os.path.join(models_dir, 'causal_model.joblib')
            output_file = os.path.join(predictions_dir, 'causal_predictions_cf.json')
            cmd = [sys.executable, 'predict_causal.py',
                   '--slices_dir', os.environ['SLICES_DIR'],
                   '--model_path', model_path,
                   '--out_path', output_file]
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"{model.upper()} prediction failed: {result.stderr}")
            continue
        
        prediction_files[model] = output_file
        print(f"{model.upper()} predictions completed successfully")
    
    return prediction_files

def merge_predictions(prediction_files, output_dir):
    """Merge predictions from all models"""
    print("Step 3: Merging predictions...")
    
    merged_predictions = []
    
    for model, file_path in prediction_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    predictions = json.load(f)
                    for pred in predictions:
                        pred['model_type'] = model
                        pred['slicer_type'] = 'cf'  # Mark as CFG Builder
                    merged_predictions.extend(predictions)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Save merged predictions
    merged_file = os.path.join(output_dir, 'merged_predictions_cf.json')
    with open(merged_file, 'w') as f:
        json.dump(merged_predictions, f, indent=2)
    
    print(f"Merged predictions saved to: {merged_file}")
    return merged_file

def place_annotations(project_root, predictions_file, output_dir):
    """Place annotations using the predictions"""
    print("Step 4: Placing annotations...")
    
    cmd = [
        sys.executable, 'place_annotations.py',
        '--project_root', project_root,
        '--predictions_file', predictions_file,
        '--output_dir', output_dir,
        '--perfect_placement'  # Use perfect placement by default
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Annotation placement failed: {result.stderr}")
        return False
    
    print("Annotation placement completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run predictions using CFG Builder slicer')
    parser.add_argument('--project_root', required=True,
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', required=True,
                       help='Path to warnings file (e.g., index1.out)')
    parser.add_argument('--models', nargs='+', default=['hgt', 'gbt', 'causal'],
                       choices=['hgt', 'gbt', 'causal'],
                       help='Models to use for prediction')
    parser.add_argument('--output_dir', default='/home/ubuntu/CFWR/prediction_results_cf',
                       help='Output directory for results')
    parser.add_argument('--skip_annotation_placement', action='store_true',
                       help='Skip annotation placement step')
    
    args = parser.parse_args()
    
    # Setup environment for CFG Builder
    setup_cfg_builder_environment()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run CFG Builder slicing
    if not run_cfg_builder_slicing(args.project_root, args.warnings_file):
        print("Failed to run CFG Builder slicing")
        return 1
    
    # Run predictions
    prediction_files = run_predictions(args.models)
    if not prediction_files:
        print("No predictions were generated")
        return 1
    
    # Merge predictions
    merged_predictions_file = merge_predictions(prediction_files, args.output_dir)
    
    # Place annotations (unless skipped)
    if not args.skip_annotation_placement:
        if not place_annotations(args.project_root, merged_predictions_file, args.output_dir):
            print("Failed to place annotations")
            return 1
    
    print("Prediction with CFG Builder completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
