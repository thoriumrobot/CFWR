#!/usr/bin/env python3
"""
Focused Hyperparameter Search for Enhanced Causal Model
Tests the enhanced causal model with various configurations across all annotation types.
"""

import subprocess
import json
import os
import logging
from datetime import datetime
import itertools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_enhanced_causal_hyperparameter_search():
    """Run hyperparameter search specifically for enhanced causal model"""
    
    # Enhanced causal model parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'hidden_dim': [128, 256, 512],
        'dropout_rate': [0.1, 0.2, 0.3],
        'episodes': [20, 50, 100]
    }
    
    # Annotation types
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    # Results storage
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'enhanced_causal',
            'total_tests': 0,
            'completed_tests': 0
        }
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    logger.info(f"🔍 Starting Enhanced Causal Hyperparameter Search")
    logger.info(f"📊 Total combinations: {len(combinations)}")
    logger.info(f"📊 Annotation types: {len(annotation_types)}")
    logger.info(f"📊 Total tests: {len(combinations) * len(annotation_types)}")
    
    for annotation_type in annotation_types:
        logger.info(f"\n🎯 Testing {annotation_type.upper()} annotation type")
        results[annotation_type] = []
        
        for i, combination in enumerate(combinations):
            logger.info(f"  Testing combination {i+1}/{len(combinations)} for {annotation_type}")
            
            # Build command
            script_map = {
                'positive': 'annotation_type_rl_positive.py',
                'nonnegative': 'annotation_type_rl_nonnegative.py',
                'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
            }
            
            cmd = [
                'python', script_map[annotation_type],
                '--base_model', 'enhanced_causal',
                '--project_root', '/home/ubuntu/checker-framework/checker/tests/index',
                '--device', 'cpu'
            ]
            
            # Add hyperparameters
            params = dict(zip(param_names, combination))
            for param, value in params.items():
                cmd.extend([f'--{param}', str(value)])
            
            try:
                logger.info(f"    Running: {' '.join(cmd[-8:])}")  # Show last 8 args
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    # Parse output
                    output = result.stdout
                    final_reward = None
                    total_predictions = None
                    
                    for line in output.split('\n'):
                        if 'Episode completed:' in line and 'reward=' in line:
                            try:
                                reward_part = line.split('reward=')[1].split(',')[0]
                                final_reward = float(reward_part)
                            except:
                                pass
                        
                        if 'predictions=' in line:
                            try:
                                pred_part = line.split('predictions=')[1].strip()
                                total_predictions = int(pred_part)
                            except:
                                pass
                    
                    score = 0.0
                    if final_reward is not None and total_predictions is not None:
                        score = final_reward * total_predictions
                    
                    test_result = {
                        'params': params,
                        'final_reward': final_reward,
                        'total_predictions': total_predictions,
                        'score': score,
                        'status': 'success'
                    }
                    
                    logger.info(f"    ✅ Success: reward={final_reward}, predictions={total_predictions}, score={score:.4f}")
                    
                else:
                    test_result = {
                        'params': params,
                        'final_reward': None,
                        'total_predictions': None,
                        'score': 0.0,
                        'status': 'failed',
                        'error': result.stderr
                    }
                    logger.error(f"    ❌ Failed: {result.stderr[:100]}...")
                
                results[annotation_type].append(test_result)
                results['metadata']['completed_tests'] += 1
                
            except subprocess.TimeoutExpired:
                test_result = {
                    'params': params,
                    'final_reward': None,
                    'total_predictions': None,
                    'score': 0.0,
                    'status': 'timeout'
                }
                logger.error(f"    ⏰ Timeout")
                results[annotation_type].append(test_result)
                results['metadata']['completed_tests'] += 1
                
            except Exception as e:
                test_result = {
                    'params': params,
                    'final_reward': None,
                    'total_predictions': None,
                    'score': 0.0,
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"    💥 Error: {e}")
                results[annotation_type].append(test_result)
                results['metadata']['completed_tests'] += 1
    
    # Update metadata
    results['metadata']['total_tests'] = len(combinations) * len(annotation_types)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_causal_hyperparameter_search_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Enhanced causal hyperparameter search completed. Results saved to {results_file}")
    
    # Print summary
    print_enhanced_causal_summary(results)
    
    return results

def print_enhanced_causal_summary(results):
    """Print summary of enhanced causal hyperparameter search results"""
    
    print("\n" + "="*70)
    print("🚀 ENHANCED CAUSAL MODEL HYPERPARAMETER SEARCH SUMMARY")
    print("="*70)
    
    metadata = results['metadata']
    print(f"📊 Total Tests: {metadata['total_tests']}")
    print(f"✅ Completed Tests: {metadata['completed_tests']}")
    print(f"📈 Completion Rate: {metadata['completed_tests']/metadata['total_tests']:.1%}")
    print(f"🕒 Timestamp: {metadata['timestamp']}")
    
    # Results by annotation type
    for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
        if annotation_type in results:
            print(f"\n🎯 {annotation_type.upper()} ANNOTATION TYPE:")
            print("-" * 50)
            
            # Filter successful results
            successful_results = [r for r in results[annotation_type] if r['status'] == 'success' and r['score'] > 0]
            
            if successful_results:
                # Sort by score
                successful_results.sort(key=lambda x: x['score'], reverse=True)
                
                print(f"✅ Successful tests: {len(successful_results)}")
                print(f"📊 Top 5 configurations:")
                
                for i, result in enumerate(successful_results[:5], 1):
                    params = result['params']
                    print(f"  {i}. Score: {result['score']:>8.4f} "
                          f"LR: {params['learning_rate']:>6.3f} "
                          f"HD: {params['hidden_dim']:>3d} "
                          f"DR: {params['dropout_rate']:>4.1f} "
                          f"EP: {params['episodes']:>3d}")
                
                # Best configuration
                best = successful_results[0]
                print(f"\n🏆 BEST CONFIGURATION:")
                print(f"   Score: {best['score']:.4f}")
                print(f"   Reward: {best['final_reward']}")
                print(f"   Predictions: {best['total_predictions']}")
                print(f"   Parameters:")
                for param, value in best['params'].items():
                    print(f"     --{param} {value}")
                
                # Command for best configuration
                script_map = {
                    'positive': 'annotation_type_rl_positive.py',
                    'nonnegative': 'annotation_type_rl_nonnegative.py',
                    'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
                }
                
                cmd = [
                    'python', script_map[annotation_type],
                    '--base_model', 'enhanced_causal',
                    '--project_root', '/home/ubuntu/checker-framework/checker/tests/index'
                ]
                
                for param, value in best['params'].items():
                    cmd.extend([f'--{param}', str(value)])
                
                print(f"\n💻 Command for best configuration:")
                print(f"   {' '.join(cmd)}")
                
            else:
                print("❌ No successful configurations found")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    logger.info("🧪 Starting Enhanced Causal Model Hyperparameter Search")
    results = run_enhanced_causal_hyperparameter_search()
    logger.info("🎉 Enhanced causal hyperparameter search completed!")
