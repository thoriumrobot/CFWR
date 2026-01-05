#!/usr/bin/env python3
"""
Comprehensive test script for all annotation type models
Tests all base models and saves predictions for manual inspection
"""

import subprocess
import json
import os
import logging
from datetime import datetime
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_annotation_models():
    """Test all annotation type models and save predictions"""
    
    # Base models to test
    base_models = ['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']
    
    # Annotation types
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    # Optimal hyperparameters based on testing
    optimal_params = {
        'enhanced_causal': {
            'learning_rate': 0.001,
            'hidden_dim': 256,
            'dropout_rate': 0.3,
            'episodes': 20
        },
        'causal': {
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'episodes': 20
        },
        'gcn': {
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'episodes': 20
        },
        'hgt': {
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'episodes': 20
        },
        'gcsn': {
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'episodes': 20
        },
        'dg2n': {
            'learning_rate': 0.001,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'episodes': 20
        },
        'gbt': {
            'learning_rate': 0.001,
            'n_estimators': 100,
            'max_depth': 3,
            'min_samples_split': 2,
            'episodes': 20
        }
    }
    
    # Results storage
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(base_models) * len(annotation_types),
            'tested_models': []
        }
    }
    
    # Create predictions directory
    predictions_dir = 'annotation_predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    total_tests = len(base_models) * len(annotation_types)
    completed_tests = 0
    
    logger.info(f"🧪 Testing all annotation type models ({total_tests} total)")
    
    for annotation_type in annotation_types:
        logger.info(f"\n🎯 Testing {annotation_type.upper()} annotation type")
        results[annotation_type] = {}
        
        for base_model in base_models:
            model_name = f"{annotation_type}_{base_model}"
            logger.info(f"  Testing {model_name}")
            
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
                '--device', 'cpu'
            ]
            
            # Add optimal hyperparameters
            params = optimal_params[base_model]
            for param, value in params.items():
                cmd.extend([f'--{param}', str(value)])
            
            start_time = time.time()
            
            try:
                logger.info(f"    Running: {' '.join(cmd[-8:])}")  # Show last 8 args
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                execution_time = time.time() - start_time
                completed_tests += 1
                
                if result.returncode == 0:
                    # Parse output for metrics
                    output = result.stdout
                    
                    # Extract metrics
                    final_reward = None
                    total_predictions = None
                    accuracy = None
                    episode_rewards = []
                    
                    for line in output.split('\n'):
                        if 'Episode completed:' in line and 'reward=' in line:
                            try:
                                reward_part = line.split('reward=')[1].split(',')[0]
                                episode_reward = float(reward_part)
                                episode_rewards.append(episode_reward)
                                final_reward = episode_reward  # Last episode reward
                            except:
                                pass
                        
                        if 'predictions=' in line:
                            try:
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
                    
                    # Calculate metrics
                    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
                    score = avg_reward * total_predictions if total_predictions else 0
                    
                    # Save predictions file
                    predictions_file = os.path.join(predictions_dir, f"{model_name}_predictions.json")
                    predictions_data = {
                        'model_name': model_name,
                        'annotation_type': annotation_type,
                        'base_model': base_model,
                        'parameters': params,
                        'metrics': {
                            'final_reward': final_reward,
                            'avg_reward': avg_reward,
                            'total_predictions': total_predictions,
                            'accuracy': accuracy,
                            'score': score,
                            'episode_rewards': episode_rewards
                        },
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat(),
                        'output': output
                    }
                    
                    with open(predictions_file, 'w') as f:
                        json.dump(predictions_data, f, indent=2)
                    
                    test_result = {
                        'params': params,
                        'final_reward': final_reward,
                        'avg_reward': avg_reward,
                        'total_predictions': total_predictions,
                        'accuracy': accuracy,
                        'score': score,
                        'episode_rewards': episode_rewards,
                        'execution_time': execution_time,
                        'predictions_file': predictions_file,
                        'status': 'success'
                    }
                    
                    logger.info(f"    ✅ Success: avg_reward={avg_reward:.3f}, final_reward={final_reward:.3f}, "
                              f"predictions={total_predictions}, score={score:.3f}, time={execution_time:.1f}s")
                    
                else:
                    test_result = {
                        'params': params,
                        'final_reward': None,
                        'avg_reward': None,
                        'total_predictions': None,
                        'accuracy': None,
                        'score': 0.0,
                        'episode_rewards': [],
                        'execution_time': execution_time,
                        'predictions_file': None,
                        'status': 'failed',
                        'error': result.stderr
                    }
                    logger.error(f"    ❌ Failed: {result.stderr[:100]}...")
                
                results[annotation_type][base_model] = test_result
                results['metadata']['tested_models'].append(model_name)
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                test_result = {
                    'params': params,
                    'final_reward': None,
                    'avg_reward': None,
                    'total_predictions': None,
                    'accuracy': None,
                    'score': 0.0,
                    'episode_rewards': [],
                    'execution_time': execution_time,
                    'predictions_file': None,
                    'status': 'timeout'
                }
                logger.error(f"    ⏰ Timeout after {execution_time:.1f}s")
                results[annotation_type][base_model] = test_result
                results['metadata']['tested_models'].append(model_name)
                completed_tests += 1
                
            except Exception as e:
                execution_time = time.time() - start_time
                test_result = {
                    'params': params,
                    'final_reward': None,
                    'avg_reward': None,
                    'total_predictions': None,
                    'accuracy': None,
                    'score': 0.0,
                    'episode_rewards': [],
                    'execution_time': execution_time,
                    'predictions_file': None,
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"    💥 Error: {e}")
                results[annotation_type][base_model] = test_result
                results['metadata']['tested_models'].append(model_name)
                completed_tests += 1
    
    # Update metadata
    results['metadata']['completed_tests'] = completed_tests
    results['metadata']['completion_rate'] = completed_tests / total_tests
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"all_annotation_models_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Test results saved to {results_file}")
    logger.info(f"📁 Predictions saved to {predictions_dir}/")
    
    # Print summary
    print_comprehensive_summary(results)
    
    return results

def print_comprehensive_summary(results):
    """Print comprehensive summary of all model tests"""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE ANNOTATION TYPE MODELS TEST SUMMARY")
    print("="*80)
    
    metadata = results['metadata']
    print(f"📊 Total Models: {metadata['total_models']}")
    print(f"✅ Completed Tests: {metadata['completed_tests']}")
    print(f"📈 Completion Rate: {metadata['completion_rate']:.1%}")
    print(f"🕒 Timestamp: {metadata['timestamp']}")
    
    # Collect all successful results
    all_results = []
    
    for annotation_type in ['positive', 'nonnegative', 'gtenegativeone']:
        if annotation_type in results:
            for base_model, result in results[annotation_type].items():
                if result['status'] == 'success':
                    all_results.append({
                        'model': f"{annotation_type}_{base_model}",
                        'annotation_type': annotation_type,
                        'base_model': base_model,
                        'avg_reward': result['avg_reward'],
                        'final_reward': result['final_reward'],
                        'total_predictions': result['total_predictions'],
                        'score': result['score'],
                        'execution_time': result['execution_time']
                    })
    
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 TOP 10 MODELS BY COMPOSITE SCORE:")
    print("-" * 80)
    for i, result in enumerate(all_results[:10], 1):
        print(f"{i:2d}. {result['model']:<25} Score: {result['score']:>8.4f} "
              f"AvgReward: {result['avg_reward']:>6.3f} FinalReward: {result['final_reward']:>6.3f} "
              f"Pred: {result['total_predictions']:>3d} Time: {result['execution_time']:>5.1f}s")
    
    # Enhanced causal model performance
    enhanced_causal_results = [m for m in all_results if 'enhanced_causal' in m['model']]
    if enhanced_causal_results:
        print(f"\n🚀 ENHANCED CAUSAL MODEL PERFORMANCE:")
        print("-" * 80)
        for result in enhanced_causal_results:
            print(f"   {result['model']:<25} Score: {result['score']:>8.4f} "
                  f"AvgReward: {result['avg_reward']:>6.3f} FinalReward: {result['final_reward']:>6.3f} "
                  f"Pred: {result['total_predictions']:>3d}")
    
    # Model type comparison
    print(f"\n📊 MODEL TYPE COMPARISON:")
    print("-" * 80)
    
    model_types = {}
    for result in all_results:
        base_model = result['base_model']
        if base_model not in model_types:
            model_types[base_model] = []
        model_types[base_model].append(result)
    
    for base_model in sorted(model_types.keys()):
        models = model_types[base_model]
        avg_score = sum(m['score'] for m in models) / len(models)
        avg_reward = sum(m['avg_reward'] for m in models) / len(models)
        print(f"   {base_model:<18} Avg Score: {avg_score:>8.4f} "
              f"Avg Reward: {avg_reward:>6.3f} ({len(models)} models)")
    
    # Annotation type comparison
    print(f"\n🎯 ANNOTATION TYPE COMPARISON:")
    print("-" * 80)
    
    annotation_types = {}
    for result in all_results:
        annotation_type = result['annotation_type']
        if annotation_type not in annotation_types:
            annotation_types[annotation_type] = []
        annotation_types[annotation_type].append(result)
    
    for annotation_type in sorted(annotation_types.keys()):
        models = annotation_types[annotation_type]
        avg_score = sum(m['score'] for m in models) / len(models)
        avg_reward = sum(m['avg_reward'] for m in models) / len(models)
        print(f"   {annotation_type:<18} Avg Score: {avg_score:>8.4f} "
              f"Avg Reward: {avg_reward:>6.3f} ({len(models)} models)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    logger.info("🧪 Starting Comprehensive Test of All Annotation Type Models")
    results = test_all_annotation_models()
    logger.info("🎉 Comprehensive test completed!")
