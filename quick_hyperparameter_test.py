#!/usr/bin/env python3
"""
Quick Hyperparameter Test for Enhanced Causal Model
Tests the enhanced causal model with a few key configurations to verify it works.
"""

import subprocess
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_causal_configurations():
    """Test enhanced causal model with key configurations"""
    
    # Test configurations for enhanced causal model
    test_configs = [
        {
            'annotation_type': 'positive',
            'base_model': 'enhanced_causal',
            'params': {
                'learning_rate': 0.001,
                'hidden_dim': 256,
                'dropout_rate': 0.3,
                'episodes': 20
            }
        },
        {
            'annotation_type': 'nonnegative',
            'base_model': 'enhanced_causal',
            'params': {
                'learning_rate': 0.005,
                'hidden_dim': 256,
                'dropout_rate': 0.2,
                'episodes': 20
            }
        },
        {
            'annotation_type': 'gtenegativeone',
            'base_model': 'enhanced_causal',
            'params': {
                'learning_rate': 0.001,
                'hidden_dim': 256,
                'dropout_rate': 0.3,
                'episodes': 20
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"🧪 Testing {config['annotation_type']} with enhanced_causal model")
        
        # Build command
        script_map = {
            'positive': 'annotation_type_rl_positive.py',
            'nonnegative': 'annotation_type_rl_nonnegative.py',
            'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
        }
        
        cmd = [
            'python', script_map[config['annotation_type']],
            '--base_model', config['base_model'],
            '--project_root', '/home/ubuntu/checker-framework/checker/tests/index',
            '--device', 'cpu'
        ]
        
        # Add hyperparameters
        for param, value in config['params'].items():
            cmd.extend([f'--{param}', str(value)])
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse output
                output = result.stdout
                final_reward = None
                total_predictions = None
                accuracy = None
                
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
                    
                    if 'Accuracy:' in line:
                        try:
                            acc_part = line.split('Accuracy:')[1].strip()
                            accuracy = float(acc_part)
                        except:
                            pass
                
                score = 0.0
                if final_reward is not None and total_predictions is not None:
                    if accuracy is not None:
                        score = final_reward * total_predictions * accuracy
                    else:
                        score = final_reward * total_predictions
                
                test_result = {
                    'annotation_type': config['annotation_type'],
                    'base_model': config['base_model'],
                    'params': config['params'],
                    'final_reward': final_reward,
                    'total_predictions': total_predictions,
                    'accuracy': accuracy,
                    'score': score,
                    'status': 'success'
                }
                
                logger.info(f"✅ Success: reward={final_reward}, predictions={total_predictions}, accuracy={accuracy}, score={score:.4f}")
                
            else:
                test_result = {
                    'annotation_type': config['annotation_type'],
                    'base_model': config['base_model'],
                    'params': config['params'],
                    'final_reward': None,
                    'total_predictions': None,
                    'accuracy': None,
                    'score': 0.0,
                    'status': 'failed',
                    'error': result.stderr
                }
                logger.error(f"❌ Failed: {result.stderr}")
            
            results.append(test_result)
            
        except subprocess.TimeoutExpired:
            test_result = {
                'annotation_type': config['annotation_type'],
                'base_model': config['base_model'],
                'params': config['params'],
                'final_reward': None,
                'total_predictions': None,
                'accuracy': None,
                'score': 0.0,
                'status': 'timeout'
            }
            logger.error(f"⏰ Timeout for {config['annotation_type']}")
            results.append(test_result)
            
        except Exception as e:
            test_result = {
                'annotation_type': config['annotation_type'],
                'base_model': config['base_model'],
                'params': config['params'],
                'final_reward': None,
                'total_predictions': None,
                'accuracy': None,
                'score': 0.0,
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"💥 Error for {config['annotation_type']}: {e}")
            results.append(test_result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_causal_quick_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Quick test results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("🚀 ENHANCED CAUSAL MODEL QUICK TEST SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in results if r['status'] == 'success']
    
    print(f"📊 Total Tests: {len(results)}")
    print(f"✅ Successful: {len(successful_tests)}")
    print(f"❌ Failed: {len(results) - len(successful_tests)}")
    print(f"📈 Success Rate: {len(successful_tests)/len(results):.1%}")
    
    if successful_tests:
        print(f"\n🏆 SUCCESSFUL TESTS:")
        print("-" * 60)
        for result in successful_tests:
            reward_str = f"{result['final_reward']:>6.3f}" if result['final_reward'] is not None else "   N/A"
            pred_str = f"{result['total_predictions']:>3d}" if result['total_predictions'] is not None else " N/A"
            acc_str = f"{result['accuracy']:>5.3f}" if result['accuracy'] is not None else "  N/A"
            print(f"   {result['annotation_type']:<15} Score: {result['score']:>8.4f} "
                  f"Reward: {reward_str} Pred: {pred_str} "
                  f"Acc: {acc_str}")
        
        avg_score = sum(r['score'] for r in successful_tests) / len(successful_tests)
        print(f"\n📊 Average Score: {avg_score:.4f}")
    
    print("\n" + "="*60)
    
    return results

if __name__ == "__main__":
    logger.info("🧪 Starting Quick Test for Enhanced Causal Model")
    results = test_enhanced_causal_configurations()
    logger.info("🎉 Quick test completed!")
