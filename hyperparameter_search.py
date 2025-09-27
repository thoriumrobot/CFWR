#!/usr/bin/env python3
"""
Hyperparameter Search Script for All CFWR Models
Tests different hyperparameters for HGT, GBT, Causal, GCN, GCSN, and DG2N models
"""

import os
import json
import subprocess
import argparse
import time
import logging
from itertools import product
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterSearch:
    def __init__(self, warnings_file='index1.out', project_root='/home/ubuntu/checker-framework/checker/tests/index'):
        self.warnings_file = warnings_file
        self.project_root = project_root
        self.results = {}
        
        # Define hyperparameter grids for each model
        self.hyperparameter_grids = {
            'hgt': {
                'learning_rate': [0.001, 0.005, 0.01],
                'episodes': [10, 20, 50],
                'hidden_dim': [128, 256, 512],
                'dropout_rate': [0.1, 0.3, 0.5]
            },
            'gbt': {
                'learning_rate': [0.05, 0.1, 0.2],
                'episodes': [10, 20, 50],
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'causal': {
                'learning_rate': [0.001, 0.005, 0.01],
                'episodes': [10, 20, 50],
                'hidden_dim': [128, 256, 512],
                'dropout_rate': [0.1, 0.3, 0.5]
            },
            'gcn': {
                'learning_rate': [0.001, 0.005, 0.01],
                'episodes': [10, 20, 50],
                'hidden_dim': [64, 128, 256],
                'dropout_rate': [0.1, 0.3, 0.5]
            },
            'gcsn': {
                'learning_rate': [0.001, 0.005, 0.01],
                'episodes': [10, 20, 50],
                'hidden_dim': [128, 256, 512],
                'dropout_rate': [0.1, 0.3, 0.5]
            },
            'dg2n': {
                'learning_rate': [0.001, 0.005, 0.01],
                'episodes': [10, 20, 50],
                'hidden_dim': [128, 256, 512],
                'dropout_rate': [0.1, 0.3, 0.5]
            }
        }
    
    def run_model_training(self, model_name, params):
        """Run training for a specific model with given parameters"""
        logger.info(f"Training {model_name} with params: {params}")
        
        # Build command
        script_path = f"binary_rl_{model_name}_standalone.py"
        cmd = [
            "python", script_path,
            "--warnings_file", self.warnings_file,
            "--project_root", self.project_root,
            "--episodes", str(params['episodes']),
            "--learning_rate", str(params['learning_rate'])
        ]
        
        # Add model-specific parameters
        if model_name in ['hgt', 'causal', 'gcn', 'gcsn', 'dg2n']:
            if 'hidden_dim' in params:
                cmd.extend(["--hidden_dim", str(params['hidden_dim'])])
            if 'dropout_rate' in params:
                cmd.extend(["--dropout_rate", str(params['dropout_rate'])])
        
        if model_name == 'gbt':
            if 'n_estimators' in params:
                cmd.extend(["--n_estimators", str(params['n_estimators'])])
            if 'max_depth' in params:
                cmd.extend(["--max_depth", str(params['max_depth'])])
            if 'min_samples_split' in params:
                cmd.extend(["--min_samples_split", str(params['min_samples_split'])])
        
        try:
            # Run training
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            end_time = time.time()
            
            training_time = end_time - start_time
            
            if result.returncode == 0:
                # Parse training output to extract metrics
                output_lines = result.stdout.split('\n')
                final_reward = None
                total_predictions = 0
                
                for line in output_lines:
                    if 'Episode completed: reward=' in line:
                        try:
                            reward_str = line.split('reward=')[1].split(',')[0]
                            final_reward = float(reward_str)
                        except:
                            pass
                    if 'predictions=' in line:
                        try:
                            pred_str = line.split('predictions=')[1]
                            total_predictions = int(pred_str)
                        except:
                            pass
                
                # Calculate score (higher is better)
                score = 0
                if final_reward is not None:
                    score += final_reward * 100  # Weight reward heavily
                if total_predictions > 0:
                    score += total_predictions * 10  # Reward making predictions
                score += max(0, 300 - training_time) / 10  # Reward efficiency
                
                return {
                    'success': True,
                    'score': score,
                    'final_reward': final_reward,
                    'total_predictions': total_predictions,
                    'training_time': training_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'score': 0,
                    'error': result.stderr,
                    'stdout': result.stdout,
                    'training_time': training_time
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'score': 0,
                'error': 'Training timeout',
                'training_time': 300
            }
        except Exception as e:
            return {
                'success': False,
                'score': 0,
                'error': str(e),
                'training_time': 0
            }
    
    def search_model_hyperparameters(self, model_name, max_combinations=20):
        """Search hyperparameters for a specific model"""
        logger.info(f"Starting hyperparameter search for {model_name}")
        
        if model_name not in self.hyperparameter_grids:
            logger.error(f"No hyperparameter grid defined for {model_name}")
            return
        
        grid = self.hyperparameter_grids[model_name]
        
        # Generate parameter combinations
        param_names = list(grid.keys())
        param_values = list(grid.values())
        combinations = list(product(*param_values))
        
        # Limit combinations for efficiency
        if len(combinations) > max_combinations:
            import random
            combinations = random.sample(combinations, max_combinations)
            logger.info(f"Limited to {max_combinations} random combinations out of {len(list(product(*param_values)))} total")
        
        model_results = []
        best_score = -float('inf')
        best_params = None
        
        for i, param_values in enumerate(combinations):
            params = dict(zip(param_names, param_values))
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")
            
            result = self.run_model_training(model_name, params)
            result['parameters'] = params
            
            model_results.append(result)
            
            if result['success'] and result['score'] > best_score:
                best_score = result['score']
                best_params = params
                logger.info(f"New best score: {best_score} with params: {best_params}")
        
        # Sort results by score
        model_results.sort(key=lambda x: x['score'], reverse=True)
        
        self.results[model_name] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': model_results
        }
        
        logger.info(f"Completed hyperparameter search for {model_name}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")
    
    def run_full_search(self, models=None, max_combinations_per_model=15):
        """Run hyperparameter search for all models"""
        if models is None:
            models = list(self.hyperparameter_grids.keys())
        
        logger.info(f"Starting full hyperparameter search for models: {models}")
        
        for model_name in models:
            try:
                self.search_model_hyperparameters(model_name, max_combinations_per_model)
            except Exception as e:
                logger.error(f"Error searching hyperparameters for {model_name}: {e}")
                self.results[model_name] = {
                    'best_params': None,
                    'best_score': -float('inf'),
                    'error': str(e)
                }
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save hyperparameter search results to file"""
        results_file = 'hyperparameter_search_results.json'
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, model_data in self.results.items():
            json_results[model_name] = {
                'best_params': model_data['best_params'],
                'best_score': model_data['best_score']
            }
            if 'all_results' in model_data:
                # Keep only top 5 results for each model to save space
                top_results = model_data['all_results'][:5]
                json_results[model_name]['top_results'] = []
                for result in top_results:
                    json_results[model_name]['top_results'].append({
                        'parameters': result['parameters'],
                        'score': result['score'],
                        'final_reward': result.get('final_reward'),
                        'total_predictions': result.get('total_predictions'),
                        'training_time': result.get('training_time'),
                        'success': result['success']
                    })
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def print_summary(self):
        """Print summary of hyperparameter search results"""
        print("\n" + "="*80)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("="*80)
        
        for model_name, model_data in self.results.items():
            print(f"\n{model_name.upper()} MODEL:")
            print("-" * 40)
            
            if model_data['best_params']:
                print(f"Best Score: {model_data['best_score']:.4f}")
                print("Best Parameters:")
                for param, value in model_data['best_params'].items():
                    print(f"  {param}: {value}")
                
                if 'all_results' in model_data and model_data['all_results']:
                    print(f"\nTop 3 Results:")
                    for i, result in enumerate(model_data['all_results'][:3]):
                        print(f"  {i+1}. Score: {result['score']:.4f}, Reward: {result.get('final_reward', 'N/A')}, "
                              f"Predictions: {result.get('total_predictions', 'N/A')}")
            else:
                print("No successful runs found")
                if 'error' in model_data:
                    print(f"Error: {model_data['error']}")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for CFWR Models')
    parser.add_argument('--warnings_file', default='index1.out', help='Path to warnings file')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index', 
                       help='Project root directory')
    parser.add_argument('--models', nargs='+', default=['hgt', 'gbt', 'causal', 'gcn', 'gcsn', 'dg2n'],
                       help='Models to search (default: all)')
    parser.add_argument('--max_combinations', type=int, default=15,
                       help='Maximum combinations to test per model (default: 15)')
    
    args = parser.parse_args()
    
    # Create hyperparameter search instance
    search = HyperparameterSearch(
        warnings_file=args.warnings_file,
        project_root=args.project_root
    )
    
    # Run search
    search.run_full_search(
        models=args.models,
        max_combinations_per_model=args.max_combinations
    )

if __name__ == "__main__":
    main()
