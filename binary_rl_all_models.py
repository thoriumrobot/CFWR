#!/usr/bin/env python3
"""
Binary Reinforcement Learning Training Script for All Models
Runs binary RL training for HGT, GBT, and Causal models to predict annotation placement.
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

def run_binary_rl_training(model_type, slices_dir, cfg_dir, episodes, learning_rate, checker_type, device='cpu'):
    """Run binary RL training for a specific model"""
    logger.info(f"Starting binary RL training for {model_type.upper()} model")
    
    # Determine the script to run
    if model_type == 'hgt':
        script_path = 'binary_rl_hgt.py'
    elif model_type == 'gbt':
        script_path = 'binary_rl_gbt.py'
    elif model_type == 'causal':
        script_path = 'binary_rl_causal.py'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Build command
    cmd = [
        'python', script_path,
        '--slices_dir', slices_dir,
        '--cfg_dir', cfg_dir,
        '--episodes', str(episodes),
        '--learning_rate', str(learning_rate),
        '--checker_type', checker_type
    ]
    
    if model_type in ['hgt', 'causal']:
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

def validate_directories(slices_dir, cfg_dir):
    """Validate that required directories exist and contain data"""
    logger.info("Validating directories...")
    
    # Check slices directory
    if not os.path.exists(slices_dir):
        logger.error(f"Slices directory does not exist: {slices_dir}")
        return False
    
    # Check for Java files in slices directory
    java_files = []
    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    if not java_files:
        logger.error(f"No Java files found in slices directory: {slices_dir}")
        return False
    
    logger.info(f"Found {len(java_files)} Java files in slices directory")
    
    # Check CFG directory
    if not os.path.exists(cfg_dir):
        logger.error(f"CFG directory does not exist: {cfg_dir}")
        return False
    
    # Check for JSON files in CFG directory
    json_files = []
    for root, dirs, files in os.walk(cfg_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        logger.error(f"No JSON CFG files found in CFG directory: {cfg_dir}")
        return False
    
    logger.info(f"Found {len(json_files)} CFG files in CFG directory")
    
    return True

def check_model_outputs():
    """Check if model outputs were generated successfully"""
    logger.info("Checking model outputs...")
    
    model_files = [
        'models/binary_rl_hgt_model.pth',
        'models/binary_rl_gbt_model.joblib',
        'models/binary_rl_causal_model.pth',
        'models/binary_rl_hgt_stats.json',
        'models/binary_rl_gbt_stats.json',
        'models/binary_rl_causal_stats.json'
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
    parser.add_argument('--slices_dir', default='slices_aug', help='Directory containing augmented slices')
    parser.add_argument('--cfg_dir', default='cfg_output', help='Directory containing CFGs')
    parser.add_argument('--episodes', type=int, default=30, help='Number of training episodes per model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checker_type', default='index', choices=['index', 'nullness'], help='Checker type')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--models', nargs='+', default=['hgt', 'gbt', 'causal'], 
                       help='Models to train (hgt, gbt, causal)')
    parser.add_argument('--skip_validation', action='store_true', help='Skip directory validation')
    
    args = parser.parse_args()
    
    logger.info("Starting binary RL training for all models")
    logger.info(f"Models: {args.models}")
    logger.info(f"Slices directory: {args.slices_dir}")
    logger.info(f"CFG directory: {args.cfg_dir}")
    logger.info(f"Episodes per model: {args.episodes}")
    
    # Validate directories
    if not args.skip_validation:
        if not validate_directories(args.slices_dir, args.cfg_dir):
            logger.error("Directory validation failed. Use --skip_validation to bypass.")
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
            slices_dir=args.slices_dir,
            cfg_dir=args.cfg_dir,
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
