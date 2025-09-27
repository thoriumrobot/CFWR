#!/usr/bin/env python3
"""
Updated Binary Reinforcement Learning Training Script for All Models
Runs binary RL training for HGT, GBT, Causal, GCN, GCSN, and DG2N models to predict annotation placement.
Uses correct project root and index1.small.out for testing.
"""

import os
import json
import argparse
import subprocess
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_binary_rl_training(model_type, project_root, warnings_file, cfwr_root, episodes, learning_rate, checker_type, device='cpu'):
    """Run binary RL training for a specific model"""
    logger.info(f"Starting binary RL training for {model_type.upper()} model")
    
    # Determine the script to run
    if model_type == 'hgt':
        script_path = 'binary_rl_hgt_standalone.py'
    elif model_type == 'gbt':
        script_path = 'binary_rl_gbt_standalone.py'
    elif model_type == 'causal':
        script_path = 'binary_rl_causal_standalone.py'
    elif model_type == 'gcn':
        script_path = 'binary_rl_gcn_standalone.py'
    elif model_type == 'gcsn':
        script_path = 'binary_rl_gcsn_standalone.py'
    elif model_type == 'dg2n':
        script_path = 'binary_rl_dg2n_standalone.py'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Build command with correct paths
    cmd = [
        'python', script_path,
        '--project_root', project_root,
        '--warnings_file', warnings_file,
        '--cfwr_root', cfwr_root,
        '--episodes', str(episodes),
        '--learning_rate', str(learning_rate),
        '--checker_type', checker_type
    ]
    
    if model_type in ['hgt', 'causal', 'gcn', 'gcsn', 'dg2n']:
        cmd.extend(['--device', device])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the training script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info(f"{model_type.upper()} binary RL training completed successfully")
            logger.info(f"Output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"{model_type.upper()} binary RL training failed")
            logger.error(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error(f"{model_type.upper()} binary RL training timed out")
        return False, "Training timed out"
    except Exception as e:
        logger.error(f"Error running {model_type.upper()} binary RL training: {e}")
        return False, str(e)

def validate_environment(project_root, warnings_file, cfwr_root):
    """Validate that required files and directories exist"""
    logger.info("Validating environment...")
    
    # Check project root
    if not os.path.exists(project_root):
        logger.error(f"Project root does not exist: {project_root}")
        return False
    
    # Check warnings file
    if not os.path.exists(warnings_file):
        logger.error(f"Warnings file does not exist: {warnings_file}")
        return False
    
    # Check CFWR root
    if not os.path.exists(cfwr_root):
        logger.error(f"CFWR root does not exist: {cfwr_root}")
        return False
    
    # Check for Java files in project root
    java_files = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    if not java_files:
        logger.error(f"No Java files found in project root: {project_root}")
        return False
    
    logger.info(f"Found {len(java_files)} Java files in project root")
    
    # Check CFWR directories
    cfwr_dirs = ['slices_aug', 'cfg_output', 'models']
    for dir_name in cfwr_dirs:
        dir_path = os.path.join(cfwr_root, dir_name)
        if not os.path.exists(dir_path):
            logger.warning(f"CFWR directory does not exist: {dir_path}")
            # Create directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    logger.info("Environment validation completed")
    return True

def check_model_outputs():
    """Check if model outputs were generated successfully"""
    logger.info("Checking model outputs...")
    
    model_files = [
        'models/binary_rl_hgt_model.pth',
        'models/binary_rl_gbt_model.joblib',
        'models/binary_rl_causal_model.pth',
        'models/binary_rl_gcn_model.pth',
        'models/binary_rl_gcsn_model.pth',
        'models/binary_rl_dg2n_model.pth',
        'models/binary_rl_hgt_stats.json',
        'models/binary_rl_gbt_stats.json',
        'models/binary_rl_causal_stats.json',
        'models/binary_rl_gcn_stats.json',
        'models/binary_rl_gcsn_stats.json',
        'models/binary_rl_dg2n_stats.json'
    ]
    
    results = {}
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            results[model_file] = {'exists': True, 'size': size}
            logger.info(f"✓ {model_file} exists ({size} bytes)")
        else:
            results[model_file] = {'exists': False, 'size': 0}
            logger.warning(f"✗ {model_file} missing")
    
    return results

def generate_summary_report(results):
    """Generate a summary report of the training results"""
    logger.info("Generating summary report...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models_trained': [],
        'model_outputs': {},
        'overall_success': True
    }
    
    # Check model training results
    for model_type, (success, output) in results.items():
        report['models_trained'].append({
            'model': model_type,
            'success': success,
            'output_length': len(output) if output else 0
        })
        
        if not success:
            report['overall_success'] = False
    
    # Check model outputs
    output_results = check_model_outputs()
    report['model_outputs'] = output_results
    
    # Save report
    report_file = 'models/binary_rl_training_report.json'
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Summary report saved to {report_file}")
    return report

def main():
    parser = argparse.ArgumentParser(description='Binary RL Training for All Models')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index', 
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/CFWR/index1.small.out', 
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/CFWR', 
                       help='Root directory of CFWR project')
    parser.add_argument('--episodes', type=int, default=20, help='Number of training episodes per model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checker_type', default='index', choices=['index', 'nullness'], help='Checker type')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--models', nargs='+', default=['hgt', 'gbt', 'causal', 'gcn', 'gcsn', 'dg2n'], 
                       help='Models to train (hgt, gbt, causal, gcn, gcsn, dg2n)')
    parser.add_argument('--skip_validation', action='store_true', help='Skip environment validation')
    
    args = parser.parse_args()
    
    logger.info("Starting binary RL training for all models")
    logger.info(f"Project root: {args.project_root}")
    logger.info(f"Warnings file: {args.warnings_file}")
    logger.info(f"CFWR root: {args.cfwr_root}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Episodes per model: {args.episodes}")
    
    # Validate environment
    if not args.skip_validation:
        if not validate_environment(args.project_root, args.warnings_file, args.cfwr_root):
            logger.error("Environment validation failed. Use --skip_validation to bypass.")
            return 1
    
    # Train each model
    results = {}
    for model_type in args.models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type.upper()} model")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        success, output = run_binary_rl_training(
            model_type=model_type,
            project_root=args.project_root,
            warnings_file=args.warnings_file,
            cfwr_root=args.cfwr_root,
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            checker_type=args.checker_type,
            device=args.device
        )
        end_time = time.time()
        
        results[model_type] = (success, output)
        
        if success:
            logger.info(f"{model_type.upper()} training completed in {end_time - start_time:.2f} seconds")
        else:
            logger.error(f"{model_type.upper()} training failed after {end_time - start_time:.2f} seconds")
    
    # Generate summary report
    logger.info(f"\n{'='*50}")
    logger.info("Generating summary report")
    logger.info(f"{'='*50}")
    
    report = generate_summary_report(results)
    
    # Print final summary
    logger.info(f"\n{'='*50}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_type, (success, output) in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{model_type.upper()}: {status}")
    
    overall_success = all(success for success, _ in results.values())
    if overall_success:
        logger.info("All models trained successfully!")
        return 0
    else:
        logger.error("Some models failed to train")
        return 1

if __name__ == '__main__':
    exit(main())
