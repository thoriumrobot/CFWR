#!/usr/bin/env python3
"""
Model Performance Evaluation Script

This script evaluates the relative performance of HGT, GBT, and Causal models
by testing them on the same dataset and comparing various metrics.
"""

import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates the performance of different models"""
    
    def __init__(self, cfg_dir: str, models_dir: str = "models"):
        self.cfg_dir = cfg_dir
        self.models_dir = models_dir
        self.results = {}
        
    def load_cfg_data(self) -> List[Dict]:
        """Load CFG data from the test dataset"""
        cfg_files = []
        for root, dirs, files in os.walk(self.cfg_dir):
            for file in files:
                if file.endswith('.json'):
                    cfg_path = os.path.join(root, file)
                    with open(cfg_path, 'r') as f:
                        cfg_data = json.load(f)
                        cfg_files.append({
                            'file': cfg_path,
                            'method': cfg_data.get('method_name', 'unknown'),
                            'data': cfg_data
                        })
        return cfg_files
    
    def evaluate_hgt_model(self, cfg_files: List[Dict]) -> Dict:
        """Evaluate HGT model performance"""
        logger.info("Evaluating HGT model...")
        start_time = time.time()
        
        try:
            from hgt import HGTModel, create_heterodata, load_cfgs
            
            # Load existing model if available
            model_path = os.path.join(self.models_dir, "hgt_model.pth")
            if os.path.exists(model_path):
                # Create metadata for HGT model
                metadata = (['node'], [('node', 'to', 'node')])
                model = HGTModel(
                    in_channels=2, 
                    hidden_channels=64, 
                    out_channels=2, 
                    num_heads=4, 
                    num_layers=2, 
                    metadata=metadata
                )
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                logger.info("Loaded existing HGT model")
            else:
                logger.info("No existing HGT model found, using random initialization")
                # Create metadata for HGT model
                metadata = (['node'], [('node', 'to', 'node')])
                model = HGTModel(
                    in_channels=2, 
                    hidden_channels=64, 
                    out_channels=2, 
                    num_heads=4, 
                    num_layers=2, 
                    metadata=metadata
                )
                model.eval()
            
            predictions = []
            processing_times = []
            
            for cfg_file in cfg_files:
                try:
                    cfg_start = time.time()
                    heterodata = create_heterodata(cfg_file['data'])
                    if heterodata is None:
                        continue
                    
                    with torch.no_grad():
                        logits = model(heterodata['node'].x)
                        probabilities = torch.softmax(logits, dim=1)
                        pred = probabilities[:, 1].mean().item()  # Average probability of annotation
                    
                    predictions.append(pred)
                    processing_times.append(time.time() - cfg_start)
                    
                except Exception as e:
                    logger.warning(f"HGT error processing {cfg_file['method']}: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            return {
                'model': 'HGT',
                'total_time': total_time,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'predictions_count': len(predictions),
                'avg_prediction': np.mean(predictions) if predictions else 0,
                'prediction_std': np.std(predictions) if predictions else 0,
                'success_rate': len(predictions) / len(cfg_files) if cfg_files else 0
            }
            
        except Exception as e:
            logger.error(f"HGT evaluation failed: {e}")
            return {
                'model': 'HGT',
                'error': str(e),
                'total_time': time.time() - start_time,
                'success_rate': 0
            }
    
    def evaluate_gbt_model(self, cfg_files: List[Dict]) -> Dict:
        """Evaluate GBT model performance"""
        logger.info("Evaluating GBT model...")
        start_time = time.time()
        
        try:
            from gbt import load_cfgs as load_cfgs_gbt, extract_features_from_cfg
            from sklearn.ensemble import GradientBoostingClassifier
            
            # Load existing model if available
            model_path = os.path.join(self.models_dir, "gbt_model.pkl")
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("Loaded existing GBT model")
            else:
                logger.info("No existing GBT model found, using default classifier")
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                # Train on dummy data to avoid errors
                X_dummy = np.random.rand(10, 8)  # 8 features as expected by GBT
                y_dummy = np.random.randint(0, 2, 10)
                model.fit(X_dummy, y_dummy)
            
            predictions = []
            processing_times = []
            
            for cfg_file in cfg_files:
                try:
                    cfg_start = time.time()
                    features = extract_features_from_cfg(cfg_file['data'])
                    if len(features) != 8:  # GBT expects 8 features
                        continue
                    
                    pred = model.predict_proba([features])[0][1]  # Probability of class 1
                    predictions.append(pred)
                    processing_times.append(time.time() - cfg_start)
                    
                except Exception as e:
                    logger.warning(f"GBT error processing {cfg_file['method']}: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            return {
                'model': 'GBT',
                'total_time': total_time,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'predictions_count': len(predictions),
                'avg_prediction': np.mean(predictions) if predictions else 0,
                'prediction_std': np.std(predictions) if predictions else 0,
                'success_rate': len(predictions) / len(cfg_files) if cfg_files else 0
            }
            
        except Exception as e:
            logger.error(f"GBT evaluation failed: {e}")
            return {
                'model': 'GBT',
                'error': str(e),
                'total_time': time.time() - start_time,
                'success_rate': 0
            }
    
    def evaluate_causal_model(self, cfg_files: List[Dict]) -> Dict:
        """Evaluate Causal model performance"""
        logger.info("Evaluating Causal model...")
        start_time = time.time()
        
        try:
            from causal_model import load_cfgs as load_cfgs_causal, extract_features_and_labels
            import torch.nn as nn
            
            # Create a simple neural network model
            class SimpleCausalModel(nn.Module):
                def __init__(self, input_dim=12, hidden_dim=64, num_classes=2):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, num_classes)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
            
            # Load existing model if available
            model_path = os.path.join(self.models_dir, "causal_model.pth")
            if os.path.exists(model_path):
                model = SimpleCausalModel(input_dim=12, hidden_dim=64, num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                logger.info("Loaded existing Causal model")
            else:
                logger.info("No existing Causal model found, using random initialization")
                model = SimpleCausalModel(input_dim=12, hidden_dim=64, num_classes=2)
                model.eval()
            
            predictions = []
            processing_times = []
            
            for cfg_file in cfg_files:
                try:
                    cfg_start = time.time()
                    features_list = extract_features_and_labels(cfg_file['data'], [])  # Empty annotations for testing
                    if not features_list:
                        continue
                    
                    # Use the first set of features
                    features = features_list[0]
                    if len(features) != 12:  # Causal model expects 12 features
                        continue
                    
                    with torch.no_grad():
                        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                        logits = model(x)
                        probabilities = torch.softmax(logits, dim=1)
                        pred = probabilities[0][1].item()  # Probability of class 1
                    
                    predictions.append(pred)
                    processing_times.append(time.time() - cfg_start)
                    
                except Exception as e:
                    logger.warning(f"Causal error processing {cfg_file['method']}: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            return {
                'model': 'Causal',
                'total_time': total_time,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'predictions_count': len(predictions),
                'avg_prediction': np.mean(predictions) if predictions else 0,
                'prediction_std': np.std(predictions) if predictions else 0,
                'success_rate': len(predictions) / len(cfg_files) if cfg_files else 0
            }
            
        except Exception as e:
            logger.error(f"Causal evaluation failed: {e}")
            return {
                'model': 'Causal',
                'error': str(e),
                'total_time': time.time() - start_time,
                'success_rate': 0
            }
    
    def run_evaluation(self) -> Dict:
        """Run comprehensive evaluation of all models"""
        logger.info("Starting model performance evaluation...")
        
        # Load test data
        cfg_files = self.load_cfg_data()
        logger.info(f"Loaded {len(cfg_files)} CFG files for evaluation")
        
        if not cfg_files:
            logger.error("No CFG files found for evaluation")
            return {}
        
        # Evaluate each model
        results = {}
        results['HGT'] = self.evaluate_hgt_model(cfg_files)
        results['GBT'] = self.evaluate_gbt_model(cfg_files)
        results['Causal'] = self.evaluate_causal_model(cfg_files)
        
        # Calculate relative performance metrics
        results['comparison'] = self._calculate_comparison_metrics(results)
        
        return results
    
    def _calculate_comparison_metrics(self, results: Dict) -> Dict:
        """Calculate relative performance metrics"""
        comparison = {}
        
        # Extract metrics for comparison
        models = ['HGT', 'GBT', 'Causal']
        metrics = ['total_time', 'avg_processing_time', 'success_rate', 'avg_prediction']
        
        for metric in metrics:
            values = {}
            for model in models:
                if model in results and metric in results[model]:
                    values[model] = results[model][metric]
            
            if values:
                if metric in ['total_time', 'avg_processing_time']:
                    # Lower is better for time metrics
                    best_model = min(values.keys(), key=lambda k: values[k])
                    comparison[f'{metric}_best'] = best_model
                    comparison[f'{metric}_ranking'] = sorted(values.keys(), key=lambda k: values[k])
                else:
                    # Higher is better for other metrics
                    best_model = max(values.keys(), key=lambda k: values[k])
                    comparison[f'{metric}_best'] = best_model
                    comparison[f'{metric}_ranking'] = sorted(values.keys(), key=lambda k: values[k], reverse=True)
        
        return comparison
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """Print a summary of the evaluation results"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE EVALUATION SUMMARY")
        print("="*60)
        
        for model_name in ['HGT', 'GBT', 'Causal']:
            if model_name in results:
                result = results[model_name]
                print(f"\n{model_name} Model:")
                print(f"  Total Time: {result.get('total_time', 0):.4f}s")
                print(f"  Avg Processing Time: {result.get('avg_processing_time', 0):.4f}s")
                print(f"  Success Rate: {result.get('success_rate', 0):.2%}")
                print(f"  Predictions Count: {result.get('predictions_count', 0)}")
                print(f"  Avg Prediction: {result.get('avg_prediction', 0):.4f}")
                print(f"  Prediction Std: {result.get('prediction_std', 0):.4f}")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
        
        if 'comparison' in results:
            print(f"\nRelative Performance:")
            comp = results['comparison']
            for key, value in comp.items():
                if key.endswith('_best'):
                    metric = key.replace('_best', '')
                    print(f"  Best {metric}: {value}")
        
        print("\n" + "="*60)

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator(
        cfg_dir="test_results/model_evaluation/cfg_output",
        models_dir="models"
    )
    
    results = evaluator.run_evaluation()
    evaluator.print_summary(results)
    evaluator.save_results(results, "test_results/model_evaluation/evaluation_results.json")

if __name__ == "__main__":
    main()
