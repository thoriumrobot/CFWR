#!/usr/bin/env python3
"""
Complete Workflow: Specimin Training + CFG Builder Prediction

This script implements the complete workflow:
1. Train models using Specimin slicer
2. Run predictions using Checker Framework CFG Builder slicer

This follows the user's requirement to use different slicers for training vs prediction.
"""

import os
import subprocess
import sys
import argparse
import time
from pathlib import Path

def print_step(step_num, description):
    """Print a formatted step header"""
    print("\n" + "="*60)
    print(f"STEP {step_num}: {description}")
    print("="*60)

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"SUCCESS: {description} completed")
    if result.stdout.strip():
        print(f"Output: {result.stdout.strip()}")
    
    return True

def setup_environment():
    """Set up environment variables"""
    print("Setting up environment variables...")
    
    # Training environment (Specimin)
    os.environ['TRAINING_SLICES_DIR'] = '/home/ubuntu/CFWR/slices_specimin'
    os.environ['TRAINING_AUGMENTED_DIR'] = '/home/ubuntu/CFWR/slices_aug_specimin'
    os.environ['TRAINING_CFG_DIR'] = '/home/ubuntu/CFWR/cfg_output_specimin'
    os.environ['TRAINING_MODELS_DIR'] = '/home/ubuntu/CFWR/models_specimin'
    
    # Prediction environment (CFG Builder)
    os.environ['PREDICTION_SLICES_DIR'] = '/home/ubuntu/CFWR/slices_cf'
    os.environ['PREDICTION_AUGMENTED_DIR'] = '/home/ubuntu/CFWR/slices_aug_cf'
    os.environ['PREDICTION_CFG_DIR'] = '/home/ubuntu/CFWR/cfg_output_cf'
    os.environ['PREDICTION_MODELS_DIR'] = '/home/ubuntu/CFWR/models_specimin'  # Use Specimin-trained models
    
    print("Environment variables set successfully")

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        os.environ['TRAINING_SLICES_DIR'],
        os.environ['TRAINING_AUGMENTED_DIR'],
        os.environ['TRAINING_CFG_DIR'],
        os.environ['TRAINING_MODELS_DIR'],
        os.environ['PREDICTION_SLICES_DIR'],
        os.environ['PREDICTION_AUGMENTED_DIR'],
        os.environ['PREDICTION_CFG_DIR']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")

def train_with_specimin(project_root, warnings_file, models):
    """Train models using Specimin slicer"""
    print_step(1, "TRAINING WITH SPECIMIN SLICER")
    
    # Set training environment
    os.environ['SLICES_DIR'] = os.environ['TRAINING_SLICES_DIR']
    os.environ['AUGMENTED_SLICES_DIR'] = os.environ['TRAINING_AUGMENTED_DIR']
    os.environ['CFG_OUTPUT_DIR'] = os.environ['TRAINING_CFG_DIR']
    os.environ['MODELS_DIR'] = os.environ['TRAINING_MODELS_DIR']
    
    # Step 1.1: Run Specimin slicing
    if not run_command([
        sys.executable, 'pipeline.py',
        '--steps', 'slice,augment,cfg',
        '--project_root', project_root,
        '--warnings_file', warnings_file,
        '--slicer', 'specimin',
        '--output_dir', '/home/ubuntu/CFWR/training_output_specimin'
    ], "Specimin slicing and augmentation"):
        return False
    
    # Step 1.2: Train models
    for model in models:
        if not run_command([
            sys.executable, f'{model}.py'
        ], f"Training {model.upper()} model with Specimin slices"):
            return False
    
    print(f"\nTraining completed successfully!")
    print(f"Models saved to: {os.environ['TRAINING_MODELS_DIR']}")
    return True

def predict_with_cfg_builder(project_root, warnings_file, models):
    """Run predictions using CFG Builder slicer"""
    print_step(2, "PREDICTION WITH CFG BUILDER SLICER")
    
    # Set prediction environment
    os.environ['SLICES_DIR'] = os.environ['PREDICTION_SLICES_DIR']
    os.environ['AUGMENTED_SLICES_DIR'] = os.environ['PREDICTION_AUGMENTED_DIR']
    os.environ['CFG_OUTPUT_DIR'] = os.environ['PREDICTION_CFG_DIR']
    os.environ['MODELS_DIR'] = os.environ['PREDICTION_MODELS_DIR']  # Use Specimin-trained models
    
    # Step 2.1: Run CFG Builder slicing
    if not run_command([
        sys.executable, 'pipeline.py',
        '--steps', 'slice,augment,cfg',
        '--project_root', project_root,
        '--warnings_file', warnings_file,
        '--slicer', 'cf',  # Use Checker Framework CFG Builder
        '--output_dir', '/home/ubuntu/CFWR/prediction_output_cf'
    ], "CFG Builder slicing and augmentation"):
        return False
    
    # Step 2.2: Run predictions
    predictions_dir = '/home/ubuntu/CFWR/predictions_cf'
    os.makedirs(predictions_dir, exist_ok=True)
    
    prediction_files = []
    
    for model in models:
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
        
        if run_command(cmd, f"Running {model.upper()} predictions with CFG Builder slices"):
            prediction_files.append(output_file)
    
    if not prediction_files:
        print("ERROR: No predictions were generated")
        return False
    
    print(f"\nPrediction completed successfully!")
    print(f"Prediction files: {prediction_files}")
    return True

def place_annotations(project_root, predictions_dir, output_dir):
    """Place annotations using the predictions"""
    print_step(3, "ANNOTATION PLACEMENT")
    
    # Find the merged predictions file or use individual files
    merged_file = os.path.join(predictions_dir, 'merged_predictions_cf.json')
    
    if not os.path.exists(merged_file):
        # Create merged predictions file
        import json
        merged_predictions = []
        
        for file_path in os.listdir(predictions_dir):
            if file_path.endswith('_predictions_cf.json'):
                full_path = os.path.join(predictions_dir, file_path)
                try:
                    with open(full_path, 'r') as f:
                        predictions = json.load(f)
                        for pred in predictions:
                            pred['slicer_type'] = 'cf'  # Mark as CFG Builder
                        merged_predictions.extend(predictions)
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
        
        with open(merged_file, 'w') as f:
            json.dump(merged_predictions, f, indent=2)
    
    # Place annotations
    if not run_command([
        sys.executable, 'place_annotations.py',
        '--project_root', project_root,
        '--predictions_file', merged_file,
        '--output_dir', output_dir,
        '--perfect_placement'
    ], "Placing annotations"):
        return False
    
    print(f"\nAnnotation placement completed successfully!")
    print(f"Annotated project saved to: {output_dir}")
    return True

def print_summary():
    """Print a summary of the workflow"""
    print("\n" + "="*60)
    print("WORKFLOW SUMMARY")
    print("="*60)
    print("✓ Training: Used Specimin slicer")
    print("✓ Prediction: Used Checker Framework CFG Builder slicer")
    print("✓ Models: Trained with Specimin slices, predicted with CFG Builder slices")
    print("✓ Annotations: Placed using CFG Builder predictions")
    print("\nThis configuration follows your requirement:")
    print("  - Specimin for training")
    print("  - CFG Builder for prediction")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Complete workflow: Specimin training + CFG Builder prediction'
    )
    parser.add_argument('--project_root', required=True,
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', required=True,
                       help='Path to warnings file (e.g., index1.out)')
    parser.add_argument('--models', nargs='+', default=['hgt', 'gbt', 'causal'],
                       choices=['hgt', 'gbt', 'causal'],
                       help='Models to train and use for prediction')
    parser.add_argument('--output_dir', default='/home/ubuntu/CFWR/final_results',
                       help='Output directory for final results')
    parser.add_argument('--skip_annotation_placement', action='store_true',
                       help='Skip annotation placement step')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step (use existing models)')
    parser.add_argument('--skip_prediction', action='store_true',
                       help='Skip prediction step')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("CFWR Workflow: Specimin Training + CFG Builder Prediction")
    print(f"Project: {args.project_root}")
    print(f"Warnings: {args.warnings_file}")
    print(f"Models: {args.models}")
    print(f"Output: {args.output_dir}")
    
    # Setup
    setup_environment()
    create_directories()
    
    # Training phase
    if not args.skip_training:
        if not train_with_specimin(args.project_root, args.warnings_file, args.models):
            print("Training phase failed")
            return 1
    else:
        print("Skipping training phase (using existing models)")
    
    # Prediction phase
    if not args.skip_prediction:
        if not predict_with_cfg_builder(args.project_root, args.warnings_file, args.models):
            print("Prediction phase failed")
            return 1
    else:
        print("Skipping prediction phase")
    
    # Annotation placement
    if not args.skip_annotation_placement:
        predictions_dir = '/home/ubuntu/CFWR/predictions_cf'
        if not place_annotations(args.project_root, predictions_dir, args.output_dir):
            print("Annotation placement failed")
            return 1
    else:
        print("Skipping annotation placement")
    
    # Summary
    print_summary()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
