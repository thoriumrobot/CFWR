#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Search for All Annotation Type Models
Tests all 21 combinations (7 base models × 3 annotation types) including the new enhanced causal model.
"""

import subprocess
import json
import os
import logging
from datetime import datetime
import itertools
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comprehensive_hyperparameter_search():
    """Run comprehensive hyperparameter search for all annotation type models including enhanced causal"""
    
    # Define parameter grids for each model type
    param_grids = {
        'gcn': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'gbt': {
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'episodes': [20, 50, 100]
        },
        'causal': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'enhanced_causal': {
            'learning_rate': [0.001, 0.005, 0.01],
            'hidden_dim': [128, 256, 512],
            'dropout_rate': [0.1, 0.2, 0.3],
            'episodes': [20, 50, 100]
        },
        'hgt': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'gcsn': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'dg2n': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        }
    }
    
    # Annotation types
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    # Results storage
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(param_grids) * len(annotation_types),
            'models_tested': [],
            'search_completed': False
        }
    }
    
    # Limit combinations for comprehensive testing
    max_combinations_per_model = 5  # Increased from 3 to get better coverage
    
    total_tests = 0
    completed_tests = 0
    
    for annotation_type in annotation_types:
        results[annotation_type] = {}
        
        for base_model in param_grids.keys():
            model_name = f"{annotation_type}_{base_model}"
            logger.info(f"🔍 Starting hyperparameter search for {model_name}")
            
            param_grid = param_grids[base_model]
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # Generate combinations
            combinations = list(itertools.product(*param_values))
            
            # Limit to max_combinations
            combinations = combinations[:max_combinations_per_model]
            total_tests += len(combinations)
            
            model_results = []
            
            for i, combination in enumerate(combinations):
                logger.info(f"  Testing combination {i+1}/{len(combinations)} for {model_name}")
                
                # Build command
                script_map = {
                    'positive': 'annotation_type_rl_positive.py',
                    'nonnegative': 'annotation_type_rl_nonnegative.py',
                    'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
                }
                
                cmd = [
                    'python', script_map[annotation_type],
                    '--base_model', base_model,
                    '--project_root', '/home/ubuntu/checker-framework/checker/tests/index',
                    '--device', 'cpu'  # Ensure consistent device usage
                ]
                
                # Add hyperparameters
                params = dict(zip(param_names, combination))
                for param, value in params.items():
                    cmd.extend([f'--{param}', str(value)])
                
                start_time = time.time()
                
                try:
                    # Increased timeout for enhanced causal model
                    timeout = 600 if base_model == 'enhanced_causal' else 300
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                    
                    execution_time = time.time() - start_time
                    completed_tests += 1
                    
                    if result.returncode == 0:
                        # Parse output for metrics
                        output = result.stdout
                        
                        # Extract final reward and predictions from output
                        final_reward = None
                        total_predictions = None
                        accuracy = None
                        
                        for line in output.split('\n'):
                            if 'Episode completed:' in line and 'reward=' in line:
                                try:
                                    # Extract reward from line like "Episode completed: @Positive reward=0.246, predictions=3"
                                    reward_part = line.split('reward=')[1].split(',')[0]
                                    final_reward = float(reward_part)
                                except:
                                    pass
                            
                            if 'predictions=' in line:
                                try:
                                    # Extract predictions count
                                    pred_part = line.split('predictions=')[1].strip()
                                    total_predictions = int(pred_part)
                                except:
                                    pass
                            
                            if 'Accuracy:' in line:
                                try:
                                    acc_part = line.split('Accuracy:')[1].strip()
                                    accuracy = float(acc_part)
                                except:
                                    pass
                        
                        # Calculate composite score
                        score = 0.0
                        if final_reward is not None and total_predictions is not None:
                            # Weighted score: reward * predictions * accuracy (if available)
                            if accuracy is not None:
                                score = final_reward * total_predictions * accuracy
                            else:
                                score = final_reward * total_predictions
                        
                        model_results.append({
                            'params': params,
                            'final_reward': final_reward,
                            'total_predictions': total_predictions,
                            'accuracy': accuracy,
                            'score': score,
                            'execution_time': execution_time,
                            'status': 'success'
                        })
                        
                        logger.info(f"  ✅ Success: reward={final_reward}, predictions={total_predictions}, accuracy={accuracy}, score={score:.4f}, time={execution_time:.1f}s")
                        
                    else:
                        logger.error(f"  ❌ Failed: {result.stderr}")
                        model_results.append({
                            'params': params,
                            'final_reward': None,
                            'total_predictions': None,
                            'accuracy': None,
                            'score': 0.0,
                            'execution_time': execution_time,
                            'status': 'failed',
                            'error': result.stderr
                        })
                        
                except subprocess.TimeoutExpired:
                    execution_time = time.time() - start_time
                    logger.error(f"  ⏰ Timeout for combination {i+1} after {execution_time:.1f}s")
                    model_results.append({
                        'params': params,
                        'final_reward': None,
                        'total_predictions': None,
                        'accuracy': None,
                        'score': 0.0,
                        'execution_time': execution_time,
                        'status': 'timeout'
                    })
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"  💥 Error for combination {i+1}: {e}")
                    model_results.append({
                        'params': params,
                        'final_reward': None,
                        'total_predictions': None,
                        'accuracy': None,
                        'score': 0.0,
                        'execution_time': execution_time,
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Find best parameters
            successful_results = [r for r in model_results if r['status'] == 'success' and r['score'] > 0]
            
            if successful_results:
                best_result = max(successful_results, key=lambda x: x['score'])
                results[annotation_type][base_model] = {
                    'best_params': best_result['params'],
                    'best_score': best_result['score'],
                    'best_reward': best_result['final_reward'],
                    'best_predictions': best_result['total_predictions'],
                    'best_accuracy': best_result['accuracy'],
                    'best_execution_time': best_result['execution_time'],
                    'successful_runs': len(successful_results),
                    'total_runs': len(model_results),
                    'success_rate': len(successful_results) / len(model_results),
                    'all_results': model_results
                }
                logger.info(f"🏆 Best for {model_name}: score={best_result['score']:.4f}, params={best_result['params']}")
            else:
                results[annotation_type][base_model] = {
                    'best_params': None,
                    'best_score': 0.0,
                    'best_reward': None,
                    'best_predictions': None,
                    'best_accuracy': None,
                    'best_execution_time': None,
                    'successful_runs': 0,
                    'total_runs': len(model_results),
                    'success_rate': 0.0,
                    'all_results': model_results
                }
                logger.warning(f"⚠️  No successful results for {model_name}")
            
            results['metadata']['models_tested'].append(model_name)
    
    # Update metadata
    results['metadata']['total_tests'] = total_tests
    results['metadata']['completed_tests'] = completed_tests
    results['metadata']['completion_rate'] = completed_tests / total_tests if total_tests > 0 else 0
    results['metadata']['search_completed'] = True
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_hyperparameter_search_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Comprehensive hyperparameter search completed. Results saved to {results_file}")
    
    # Print comprehensive summary
    print_comprehensive_summary(results)
    
    return results

def print_comprehensive_summary(results):
    """Print a comprehensive summary of hyperparameter search results"""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE HYPERPARAMETER SEARCH SUMMARY")
    print("="*80)
    
    metadata = results['metadata']
    print(f"📊 Total Models Tested: {len(metadata['models_tested'])}")
    print(f"🧪 Total Test Runs: {metadata['total_tests']}")
    print(f"✅ Completed Tests: {metadata['completed_tests']}")
    print(f"📈 Completion Rate: {metadata['completion_rate']:.1%}")
    print(f"🕒 Timestamp: {metadata['timestamp']}")
    
    # Overall rankings
    all_models = []
    for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
        if annotation_type in results:
            for base_model, result in results[annotation_type].items():
                if result['best_params']:
                    all_models.append({
                        'model': f"{annotation_type}_{base_model}",
                        'score': result['best_score'],
                        'reward': result['best_reward'],
                        'predictions': result['best_predictions'],
                        'accuracy': result['best_accuracy'],
                        'success_rate': result['success_rate']
                    })
    
    # Sort by score
    all_models.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 TOP 10 MODELS BY COMPOSITE SCORE:")
    print("-" * 80)
    for i, model in enumerate(all_models[:10], 1):
        print(f"{i:2d}. {model['model']:<25} Score: {model['score']:>8.4f} "
              f"Reward: {model['reward']:>6.3f} Pred: {model['predictions']:>3d} "
              f"Acc: {model['accuracy']:>5.3f} SR: {model['success_rate']:>5.1%}")
    
    # Enhanced causal model performance
    enhanced_causal_models = [m for m in all_models if 'enhanced_causal' in m['model']]
    if enhanced_causal_models:
        print(f"\n🚀 ENHANCED CAUSAL MODEL PERFORMANCE:")
        print("-" * 80)
        for model in enhanced_causal_models:
            print(f"   {model['model']:<25} Score: {model['score']:>8.4f} "
                  f"Reward: {model['reward']:>6.3f} Pred: {model['predictions']:>3d} "
                  f"Acc: {model['accuracy']:>5.3f}")
    
    # Model type comparisons
    print(f"\n📊 MODEL TYPE COMPARISON:")
    print("-" * 80)
    
    model_types = {}
    for model in all_models:
        base_model = model['model'].split('_')[1]
        if base_model not in model_types:
            model_types[base_model] = []
        model_types[base_model].append(model)
    
    for base_model in sorted(model_types.keys()):
        models = model_types[base_model]
        avg_score = sum(m['score'] for m in models) / len(models)
        avg_success_rate = sum(m['success_rate'] for m in models) / len(models)
        print(f"   {base_model:<18} Avg Score: {avg_score:>8.4f} "
              f"Avg Success Rate: {avg_success_rate:>5.1%} ({len(models)} models)")
    
    # Annotation type comparisons
    print(f"\n🎯 ANNOTATION TYPE COMPARISON:")
    print("-" * 80)
    
    annotation_types = {}
    for model in all_models:
        annotation_type = model['model'].split('_')[0]
        if annotation_type not in annotation_types:
            annotation_types[annotation_type] = []
        annotation_types[annotation_type].append(model)
    
    for annotation_type in sorted(annotation_types.keys()):
        models = annotation_types[annotation_type]
        avg_score = sum(m['score'] for m in models) / len(models)
        avg_success_rate = sum(m['success_rate'] for m in models) / len(models)
        print(f"   {annotation_type:<18} Avg Score: {avg_score:>8.4f} "
              f"Avg Success Rate: {avg_success_rate:>5.1%} ({len(models)} models)")
    
    print("\n" + "="*80)

def generate_best_configurations_report(results):
    """Generate a report of the best configurations for each model"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"best_configurations_report_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Best Configurations Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report contains the best hyperparameter configurations for each annotation type model.\n\n")
        
        for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
            if annotation_type in results:
                f.write(f"## {annotation_type.upper()} Annotation Type\n\n")
                
                # Sort models by score
                models = [(base_model, result) for base_model, result in results[annotation_type].items()]
                models.sort(key=lambda x: x[1]['best_score'], reverse=True)
                
                for base_model, result in models:
                    f.write(f"### {base_model.upper()} Model\n\n")
                    
                    if result['best_params']:
                        f.write(f"**Best Score:** {result['best_score']:.4f}\n")
                        f.write(f"**Best Reward:** {result['best_reward']}\n")
                        f.write(f"**Best Predictions:** {result['best_predictions']}\n")
                        f.write(f"**Best Accuracy:** {result['best_accuracy']}\n")
                        f.write(f"**Success Rate:** {result['success_rate']:.1%}\n\n")
                        
                        f.write("**Best Parameters:**\n")
                        for param, value in result['best_params'].items():
                            f.write(f"- `--{param} {value}`\n")
                        f.write("\n")
                        
                        f.write("**Command Line:**\n")
                        f.write("```bash\n")
                        script_map = {
                            'positive': 'annotation_type_rl_positive.py',
                            'nonnegative': 'annotation_type_rl_nonnegative.py',
                            'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
                        }
                        
                        cmd = [
                            'python', script_map[annotation_type],
                            '--base_model', base_model,
                            '--project_root', '/home/ubuntu/checker-framework/checker/tests/index'
                        ]
                        
                        for param, value in result['best_params'].items():
                            cmd.extend([f'--{param}', str(value)])
                        
                        f.write(' '.join(cmd) + '\n')
                        f.write("```\n\n")
                    else:
                        f.write("**Status:** No successful configurations found\n\n")
    
    logger.info(f"📄 Best configurations report saved to {report_file}")
    return report_file

if __name__ == "__main__":
    logger.info("🚀 Starting Comprehensive Hyperparameter Search for All Annotation Type Models")
    logger.info("Including the new Enhanced Causal Model (21 total models)")
    
    results = run_comprehensive_hyperparameter_search()
    
    # Generate best configurations report
    report_file = generate_best_configurations_report(results)
    
    logger.info("🎉 Comprehensive hyperparameter search completed successfully!")
    logger.info(f"📊 Results saved and report generated: {report_file}")
