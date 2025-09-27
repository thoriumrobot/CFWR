#!/usr/bin/env python3
"""
Update Model Implementations with Optimal Hyperparameters
Reads hyperparameter search results and updates model implementations
"""

import json
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterUpdater:
    def __init__(self, results_file='hyperparameter_search_results.json'):
        self.results_file = results_file
        self.results = {}
        
    def load_results(self):
        """Load hyperparameter search results"""
        if not os.path.exists(self.results_file):
            logger.error(f"Results file {self.results_file} not found")
            return False
            
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Loaded results for {len(self.results)} models")
        return True
    
    def update_model_script(self, model_name, optimal_params):
        """Update a model script with optimal hyperparameters"""
        script_file = f"binary_rl_{model_name}_standalone.py"
        
        if not os.path.exists(script_file):
            logger.error(f"Script file {script_file} not found")
            return False
        
        logger.info(f"Updating {script_file} with optimal parameters: {optimal_params}")
        
        # Read the current script
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Update default values in argument parser
        updates = []
        for param, value in optimal_params.items():
            if param == 'learning_rate':
                pattern = f"parser.add_argument('--learning_rate', type=float, default="
                new_pattern = f"parser.add_argument('--learning_rate', type=float, default={value}"
                if pattern in content:
                    # Find and replace the default value
                    start = content.find(pattern) + len(pattern)
                    end = content.find(',', start)
                    if end == -1:
                        end = content.find(')', start)
                    content = content[:start] + str(value) + content[end:]
                    updates.append(f"learning_rate: {value}")
            
            elif param == 'hidden_dim':
                pattern = f"parser.add_argument('--hidden_dim', type=int, default="
                new_pattern = f"parser.add_argument('--hidden_dim', type=int, default={value}"
                if pattern in content:
                    start = content.find(pattern) + len(pattern)
                    end = content.find(',', start)
                    if end == -1:
                        end = content.find(')', start)
                    content = content[:start] + str(value) + content[end:]
                    updates.append(f"hidden_dim: {value}")
            
            elif param == 'dropout_rate':
                pattern = f"parser.add_argument('--dropout_rate', type=float, default="
                new_pattern = f"parser.add_argument('--dropout_rate', type=float, default={value}"
                if pattern in content:
                    start = content.find(pattern) + len(pattern)
                    end = content.find(',', start)
                    if end == -1:
                        end = content.find(')', start)
                    content = content[:start] + str(value) + content[end:]
                    updates.append(f"dropout_rate: {value}")
            
            elif param == 'episodes':
                pattern = f"parser.add_argument('--episodes', type=int, default="
                new_pattern = f"parser.add_argument('--episodes', type=int, default={value}"
                if pattern in content:
                    start = content.find(pattern) + len(pattern)
                    end = content.find(',', start)
                    if end == -1:
                        end = content.find(')', start)
                    content = content[:start] + str(value) + content[end:]
                    updates.append(f"episodes: {value}")
            
            elif param == 'n_estimators':
                pattern = f"parser.add_argument('--n_estimators', type=int, default="
                new_pattern = f"parser.add_argument('--n_estimators', type=int, default={value}"
                if pattern in content:
                    start = content.find(pattern) + len(pattern)
                    end = content.find(',', start)
                    if end == -1:
                        end = content.find(')', start)
                    content = content[:start] + str(value) + content[end:]
                    updates.append(f"n_estimators: {value}")
            
            elif param == 'max_depth':
                pattern = f"parser.add_argument('--max_depth', type=int, default="
                new_pattern = f"parser.add_argument('--max_depth', type=int, default={value}"
                if pattern in content:
                    start = content.find(pattern) + len(pattern)
                    end = content.find(',', start)
                    if end == -1:
                        end = content.find(')', start)
                    content = content[:start] + str(value) + content[end:]
                    updates.append(f"max_depth: {value}")
            
            elif param == 'min_samples_split':
                pattern = f"parser.add_argument('--min_samples_split', type=int, default="
                new_pattern = f"parser.add_argument('--min_samples_split', type=int, default={value}"
                if pattern in content:
                    start = content.find(pattern) + len(pattern)
                    end = content.find(',', start)
                    if end == -1:
                        end = content.find(')', start)
                    content = content[:start] + str(value) + content[end:]
                    updates.append(f"min_samples_split: {value}")
        
        # Update trainer initialization defaults
        if model_name == 'gbt':
            # Update GBT trainer defaults
            for param, value in optimal_params.items():
                if param == 'learning_rate':
                    pattern = "def __init__(self, learning_rate="
                    if pattern in content:
                        start = content.find(pattern) + len(pattern)
                        end = content.find(',', start)
                        if end == -1:
                            end = content.find(')', start)
                        content = content[:start] + str(value) + content[end:]
                
                elif param == 'n_estimators':
                    pattern = "n_estimators="
                    if pattern in content:
                        start = content.find(pattern) + len(pattern)
                        end = content.find(',', start)
                        if end == -1:
                            end = content.find(')', start)
                        content = content[:start] + str(value) + content[end:]
                
                elif param == 'max_depth':
                    pattern = "max_depth="
                    if pattern in content:
                        start = content.find(pattern) + len(pattern)
                        end = content.find(',', start)
                        if end == -1:
                            end = content.find(')', start)
                        content = content[:start] + str(value) + content[end:]
                
                elif param == 'min_samples_split':
                    pattern = "min_samples_split="
                    if pattern in content:
                        start = content.find(pattern) + len(pattern)
                        end = content.find(',', start)
                        if end == -1:
                            end = content.find(')', start)
                        content = content[:start] + str(value) + content[end:]
        
        else:
            # Update neural network trainer defaults
            for param, value in optimal_params.items():
                if param == 'learning_rate':
                    pattern = "def __init__(self, learning_rate="
                    if pattern in content:
                        start = content.find(pattern) + len(pattern)
                        end = content.find(',', start)
                        if end == -1:
                            end = content.find(')', start)
                        content = content[:start] + str(value) + content[end:]
                
                elif param == 'hidden_dim':
                    pattern = "hidden_dim="
                    if pattern in content:
                        start = content.find(pattern) + len(pattern)
                        end = content.find(',', start)
                        if end == -1:
                            end = content.find(')', start)
                        content = content[:start] + str(value) + content[end:]
                
                elif param == 'dropout_rate':
                    pattern = "dropout_rate="
                    if pattern in content:
                        start = content.find(pattern) + len(pattern)
                        end = content.find(',', start)
                        if end == -1:
                            end = content.find(')', start)
                        content = content[:start] + str(value) + content[end:]
        
        # Write the updated script
        with open(script_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {script_file} with: {', '.join(updates)}")
        return True
    
    def update_all_models(self):
        """Update all models with their optimal hyperparameters"""
        if not self.load_results():
            return False
        
        success_count = 0
        for model_name, model_data in self.results.items():
            if model_data.get('best_params'):
                if self.update_model_script(model_name, model_data['best_params']):
                    success_count += 1
                else:
                    logger.error(f"Failed to update {model_name}")
            else:
                logger.warning(f"No optimal parameters found for {model_name}")
        
        logger.info(f"Successfully updated {success_count}/{len(self.results)} models")
        return success_count > 0
    
    def print_summary(self):
        """Print summary of optimal hyperparameters"""
        if not self.results:
            logger.error("No results loaded")
            return
        
        print("\n" + "="*80)
        print("OPTIMAL HYPERPARAMETERS SUMMARY")
        print("="*80)
        
        for model_name, model_data in self.results.items():
            print(f"\n{model_name.upper()} MODEL:")
            print("-" * 40)
            
            if model_data.get('best_params'):
                print(f"Best Score: {model_data['best_score']:.4f}")
                print("Optimal Parameters:")
                for param, value in model_data['best_params'].items():
                    print(f"  {param}: {value}")
            else:
                print("No optimal parameters found")

def main():
    parser = argparse.ArgumentParser(description='Update Model Implementations with Optimal Hyperparameters')
    parser.add_argument('--results_file', default='hyperparameter_search_results.json',
                       help='Path to hyperparameter search results file')
    parser.add_argument('--print_only', action='store_true',
                       help='Only print summary, do not update files')
    
    args = parser.parse_args()
    
    updater = HyperparameterUpdater(args.results_file)
    
    if args.print_only:
        updater.load_results()
        updater.print_summary()
    else:
        if updater.update_all_models():
            updater.print_summary()
            print("\nModel implementations have been updated with optimal hyperparameters!")
        else:
            print("Failed to update model implementations")

if __name__ == "__main__":
    main()
