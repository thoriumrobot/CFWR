#!/usr/bin/env python3
"""
Training Script with Specimin Slicer

This script trains models using Specimin for slicing, following the user's requirement
to use Specimin for training and CFG Builder for prediction.
"""

import os
import subprocess
import sys
import argparse

def setup_specimin_environment():
    """Set up environment variables for Specimin training"""
    # Set environment variables for Specimin-based training
    os.environ['SLICES_DIR'] = '/home/ubuntu/CFWR/slices_specimin'
    os.environ['AUGMENTED_SLICES_DIR'] = '/home/ubuntu/CFWR/slices_aug_specimin'
    os.environ['CFG_OUTPUT_DIR'] = '/home/ubuntu/CFWR/cfg_output_specimin'
    os.environ['MODELS_DIR'] = '/home/ubuntu/CFWR/models_specimin'
    
    print("Environment configured for Specimin training:")
    print(f"  SLICES_DIR: {os.environ['SLICES_DIR']}")
    print(f"  AUGMENTED_SLICES_DIR: {os.environ['AUGMENTED_SLICES_DIR']}")
    print(f"  CFG_OUTPUT_DIR: {os.environ['CFG_OUTPUT_DIR']}")
    print(f"  MODELS_DIR: {os.environ['MODELS_DIR']}")

def run_specimin_slicing(project_root, warnings_file):
    """Run slicing using Specimin"""
    print("Step 1: Running Specimin slicing...")
    
    # Create directories
    os.makedirs(os.environ['SLICES_DIR'], exist_ok=True)
    os.makedirs(os.environ['AUGMENTED_SLICES_DIR'], exist_ok=True)
    os.makedirs(os.environ['CFG_OUTPUT_DIR'], exist_ok=True)
    os.makedirs(os.environ['MODELS_DIR'], exist_ok=True)
    
    # Run pipeline with Specimin slicer
    cmd = [
        sys.executable, 'pipeline.py',
        '--steps', 'slice,augment,cfg',
        '--project_root', project_root,
        '--warnings_file', warnings_file,
        '--slicer', 'specimin',
        '--output_dir', '/home/ubuntu/CFWR/training_output_specimin'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Specimin slicing failed: {result.stderr}")
        return False
    
    print("Specimin slicing completed successfully")
    return True

def train_models():
    """Train all models using Specimin-generated slices"""
    print("Step 2: Training models with Specimin slices...")
    
    models = ['hgt', 'gbt', 'causal']
    
    for model in models:
        print(f"Training {model.upper()} model...")
        
        if model == 'hgt':
            cmd = [sys.executable, 'hgt.py']
        elif model == 'gbt':
            cmd = [sys.executable, 'gbt.py']
        elif model == 'causal':
            cmd = [sys.executable, 'causal_model.py']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"{model.upper()} training failed: {result.stderr}")
            return False
        
        print(f"{model.upper()} training completed successfully")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train models using Specimin slicer')
    parser.add_argument('--project_root', required=True, 
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', required=True,
                       help='Path to warnings file (e.g., index1.out)')
    parser.add_argument('--models', nargs='+', default=['hgt', 'gbt', 'causal'],
                       choices=['hgt', 'gbt', 'causal'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    # Setup environment for Specimin
    setup_specimin_environment()
    
    # Run Specimin slicing
    if not run_specimin_slicing(args.project_root, args.warnings_file):
        print("Failed to run Specimin slicing")
        return 1
    
    # Train models
    if not train_models():
        print("Failed to train models")
        return 1
    
    print("Training with Specimin completed successfully!")
    print(f"Models saved to: {os.environ['MODELS_DIR']}")
    print(f"CFGs saved to: {os.environ['CFG_OUTPUT_DIR']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
