#!/usr/bin/env python3
"""
Test script to run all prediction models on Java files from the agrona project.
"""

import os
import subprocess
import json
from pathlib import Path

def run_prediction_script(script_name, java_file, model_path, output_path, cfg_dir):
    """Run a prediction script and return success status."""
    try:
        cmd = [
            'python3', script_name,
            '--java_file', java_file,
            '--model_path', model_path,
            '--out_path', output_path
        ]
        
        env = os.environ.copy()
        env['CFG_OUTPUT_DIR'] = cfg_dir
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    # Configuration
    agrona_dir = Path('/home/ubuntu/original/agrona')
    cfwr_dir = Path('/home/ubuntu/CFWR')
    cfg_dir = 'cfg_test'
    predictions_dir = cfwr_dir / 'predictions_original'
    
    # Find Java files in agrona project
    java_files = list(agrona_dir.rglob('*.java'))[:5]  # Test first 5 files
    
    print(f"Found {len(java_files)} Java files to test")
    
    results = {}
    
    for java_file in java_files:
        print(f"\n=== Testing {java_file.name} ===")
        
        # Copy file to CFWR directory
        test_file = cfwr_dir / f"test_{java_file.name}"
        subprocess.run(['cp', str(java_file), str(test_file)])
        
        # Generate CFGs
        print("Generating CFGs...")
        cfg_result = subprocess.run([
            'python3', 'cfg.py',
            '--java_file', str(test_file),
            '--out_dir', cfg_dir
        ], capture_output=True, text=True)
        
        if cfg_result.returncode != 0:
            print(f"CFG generation failed: {cfg_result.stderr}")
            continue
            
        # Test HGT prediction
        print("Testing HGT prediction...")
        hgt_success, hgt_stdout, hgt_stderr = run_prediction_script(
            'predict_hgt.py', str(test_file),
            'models/best_model.pth',
            str(predictions_dir / f"{test_file.stem}_hgt.json"),
            cfg_dir
        )
        
        # Test GBT prediction
        print("Testing GBT prediction...")
        gbt_success, gbt_stdout, gbt_stderr = run_prediction_script(
            'predict_gbt.py', str(test_file),
            'models/gbt_model_1.joblib',
            str(predictions_dir / f"{test_file.stem}_gbt.json"),
            cfg_dir
        )
        
        # Test Causal prediction
        print("Testing Causal prediction...")
        causal_success, causal_stdout, causal_stderr = run_prediction_script(
            'predict_causal.py', str(test_file),
            'models/causal_clf.joblib',
            str(predictions_dir / f"{test_file.stem}_causal.json"),
            cfg_dir
        )
        
        results[java_file.name] = {
            'hgt': hgt_success,
            'gbt': gbt_success,
            'causal': causal_success
        }
        
        print(f"HGT: {'✓' if hgt_success else '✗'}")
        print(f"GBT: {'✓' if gbt_success else '✗'}")
        print(f"Causal: {'✓' if causal_success else '✗'}")
        
        # Clean up test file
        test_file.unlink()
    
    # Summary
    print("\n=== SUMMARY ===")
    total_files = len(results)
    hgt_success = sum(1 for r in results.values() if r['hgt'])
    gbt_success = sum(1 for r in results.values() if r['gbt'])
    causal_success = sum(1 for r in results.values() if r['causal'])
    
    print(f"Total files tested: {total_files}")
    print(f"HGT predictions successful: {hgt_success}/{total_files}")
    print(f"GBT predictions successful: {gbt_success}/{total_files}")
    print(f"Causal predictions successful: {causal_success}/{total_files}")
    
    # Save results
    with open(predictions_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {predictions_dir / 'test_results.json'}")

if __name__ == '__main__':
    main()
