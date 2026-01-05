#!/usr/bin/env python3
"""
Final verification script for all annotation type models
Tests all models and saves predictions with proper parsing
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

def run_model_test(annotation_type, base_model, params):
    """Run a single model test and return results"""
    
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
    
    # Add parameters
    for param, value in params.items():
        cmd.extend([f'--{param}', str(value)])
    
    try:
        logger.info(f"    Running {annotation_type}_{base_model}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse output for metrics
            output = result.stdout
            episode_rewards = []
            final_reward = None
            total_predictions = None
            accuracy = None
            
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
            
            return {
                'status': 'success',
                'final_reward': final_reward,
                'avg_reward': avg_reward,
                'total_predictions': total_predictions,
                'accuracy': accuracy,
                'score': score,
                'episode_rewards': episode_rewards,
                'output': output
            }
        else:
            return {
                'status': 'failed',
                'error': result.stderr,
                'output': result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': 'Command timed out after 300 seconds'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def verify_all_models():
    """Verify all annotation type models and save predictions"""
    
    # Test configurations
    test_configs = [
        {
            'annotation_type': 'positive',
            'base_model': 'enhanced_causal',
            'params': {'learning_rate': 0.001, 'hidden_dim': 256, 'dropout_rate': 0.3, 'episodes': 10}
        },
        {
            'annotation_type': 'positive',
            'base_model': 'causal',
            'params': {'learning_rate': 0.001, 'hidden_dim': 128, 'dropout_rate': 0.3, 'episodes': 10}
        },
        {
            'annotation_type': 'nonnegative',
            'base_model': 'enhanced_causal',
            'params': {'learning_rate': 0.001, 'hidden_dim': 256, 'dropout_rate': 0.3, 'episodes': 10}
        },
        {
            'annotation_type': 'nonnegative',
            'base_model': 'causal',
            'params': {'learning_rate': 0.001, 'hidden_dim': 128, 'dropout_rate': 0.3, 'episodes': 10}
        },
        {
            'annotation_type': 'gtenegativeone',
            'base_model': 'enhanced_causal',
            'params': {'learning_rate': 0.001, 'hidden_dim': 256, 'dropout_rate': 0.3, 'episodes': 10}
        },
        {
            'annotation_type': 'gtenegativeone',
            'base_model': 'causal',
            'params': {'learning_rate': 0.001, 'hidden_dim': 128, 'dropout_rate': 0.3, 'episodes': 10}
        }
    ]
    
    # Create predictions directory
    predictions_dir = 'final_annotation_predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_configs),
            'completed_tests': 0,
            'successful_tests': 0
        },
        'models': {}
    }
    
    logger.info(f"🧪 Starting final verification of annotation type models")
    logger.info(f"📊 Testing {len(test_configs)} configurations")
    
    for i, config in enumerate(test_configs, 1):
        model_name = f"{config['annotation_type']}_{config['base_model']}"
        logger.info(f"\n{i}/{len(test_configs)} Testing {model_name}")
        
        # Run test
        test_result = run_model_test(
            config['annotation_type'],
            config['base_model'],
            config['params']
        )
        
        # Save prediction file
        predictions_file = os.path.join(predictions_dir, f"{model_name}_predictions.json")
        prediction_data = {
            'model_name': model_name,
            'annotation_type': config['annotation_type'],
            'base_model': config['base_model'],
            'parameters': config['params'],
            'test_result': test_result,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(predictions_file, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        # Update results
        results['models'][model_name] = {
            'config': config,
            'result': test_result,
            'predictions_file': predictions_file
        }
        
        results['metadata']['completed_tests'] += 1
        if test_result['status'] == 'success':
            results['metadata']['successful_tests'] += 1
        
        # Log result
        if test_result['status'] == 'success':
            avg_reward_str = f"{test_result['avg_reward']:.3f}" if test_result['avg_reward'] is not None else "N/A"
            final_reward_str = f"{test_result['final_reward']:.3f}" if test_result['final_reward'] is not None else "N/A"
            predictions_str = str(test_result['total_predictions']) if test_result['total_predictions'] is not None else "N/A"
            score_str = f"{test_result['score']:.3f}" if test_result['score'] is not None else "N/A"
            
            logger.info(f"    ✅ Success: avg_reward={avg_reward_str}, "
                      f"final_reward={final_reward_str}, "
                      f"predictions={predictions_str}, "
                      f"score={score_str}")
        else:
            logger.error(f"    ❌ Failed: {test_result.get('error', 'Unknown error')}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"final_verification_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("🎯 FINAL ANNOTATION TYPE MODELS VERIFICATION SUMMARY")
    print("="*70)
    
    metadata = results['metadata']
    print(f"📊 Total Tests: {metadata['total_tests']}")
    print(f"✅ Completed Tests: {metadata['completed_tests']}")
    print(f"🎉 Successful Tests: {metadata['successful_tests']}")
    print(f"📈 Success Rate: {metadata['successful_tests']/metadata['total_tests']:.1%}")
    
    # Show successful results
    successful_models = []
    for model_name, model_data in results['models'].items():
        if model_data['result']['status'] == 'success':
            result = model_data['result']
            successful_models.append({
                'model': model_name,
                'avg_reward': result['avg_reward'],
                'final_reward': result['final_reward'],
                'total_predictions': result['total_predictions'],
                'score': result['score']
            })
    
    if successful_models:
        print(f"\n🏆 SUCCESSFUL MODELS:")
        print("-" * 70)
        successful_models.sort(key=lambda x: x['score'], reverse=True)
        
        for model in successful_models:
            score_str = f"{model['score']:>8.4f}" if model['score'] is not None else "    N/A"
            avg_reward_str = f"{model['avg_reward']:>6.3f}" if model['avg_reward'] is not None else "   N/A"
            final_reward_str = f"{model['final_reward']:>6.3f}" if model['final_reward'] is not None else "   N/A"
            pred_str = f"{model['total_predictions']:>3d}" if model['total_predictions'] is not None else " N/A"
            
            print(f"   {model['model']:<25} Score: {score_str} "
                  f"AvgReward: {avg_reward_str} FinalReward: {final_reward_str} "
                  f"Pred: {pred_str}")
    
    print(f"\n📁 Predictions saved to: {predictions_dir}/")
    print(f"📄 Results saved to: {results_file}")
    print("\n" + "="*70)
    
    return results

if __name__ == "__main__":
    logger.info("🔍 Starting Final Verification of All Annotation Type Models")
    results = verify_all_models()
    logger.info("🎉 Final verification completed!")
