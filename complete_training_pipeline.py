#!/usr/bin/env python3
"""
Complete Training Pipeline for Node-Level Models with RL Integration

This script provides comprehensive training, testing, and RL integration
for all refactored node-level models.
"""

import os
import json
import time
import argparse
from typing import Dict, List
import logging
from node_level_models import NodeLevelHGTModel, NodeLevelGBTModel, NodeLevelCausalModel, debug_log

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTrainingPipeline:
    """Complete training pipeline with testing and RL integration"""
    
    def __init__(self, cfg_dir: str, models_dir: str = "models_node_level"):
        self.cfg_dir = cfg_dir
        self.models_dir = models_dir
        self.results = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        debug_log(f"Initialized training pipeline: cfg_dir={cfg_dir}, models_dir={models_dir}")
    
    def load_cfg_data(self) -> List[Dict]:
        """Load CFG data from the test dataset"""
        debug_log("Loading CFG data for training...")
        
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
        
        debug_log(f"Loaded {len(cfg_files)} CFG files")
        return cfg_files
    
    def train_hgt_model(self, cfg_files: List[Dict]) -> Dict:
        """Train HGT model with comprehensive metrics"""
        debug_log("Starting HGT model training...")
        
        model = NodeLevelHGTModel()
        start_time = time.time()
        
        # Train model
        training_result = model.train_model(cfg_files, epochs=30, learning_rate=0.001)
        training_time = time.time() - start_time
        
        if training_result['success']:
            # Save model
            model_path = os.path.join(self.models_dir, "hgt_node_level.pth")
            save_success = model.save_model(model_path)
            
            # Test model
            test_results = self.test_model(model, cfg_files, "HGT")
            
            result = {
                'model': 'HGT',
                'training_time': training_time,
                'training_result': training_result,
                'model_saved': save_success,
                'model_path': model_path,
                'test_results': test_results
            }
        else:
            result = {
                'model': 'HGT',
                'training_time': training_time,
                'error': training_result.get('error', 'Training failed'),
                'success': False
            }
        
        debug_log(f"HGT training completed: {result}")
        return result
    
    def train_gbt_model(self, cfg_files: List[Dict]) -> Dict:
        """Train GBT model with comprehensive metrics"""
        debug_log("Starting GBT model training...")
        
        model = NodeLevelGBTModel()
        start_time = time.time()
        
        # Train model
        training_result = model.train_model(cfg_files)
        training_time = time.time() - start_time
        
        if training_result['success']:
            # Save model
            model_path = os.path.join(self.models_dir, "gbt_node_level.pkl")
            save_success = model.save_model(model_path)
            
            # Test model
            test_results = self.test_model(model, cfg_files, "GBT")
            
            result = {
                'model': 'GBT',
                'training_time': training_time,
                'training_result': training_result,
                'model_saved': save_success,
                'model_path': model_path,
                'test_results': test_results
            }
        else:
            result = {
                'model': 'GBT',
                'training_time': training_time,
                'error': training_result.get('error', 'Training failed'),
                'success': False
            }
        
        debug_log(f"GBT training completed: {result}")
        return result
    
    def train_causal_model(self, cfg_files: List[Dict]) -> Dict:
        """Train Causal model with comprehensive metrics"""
        debug_log("Starting Causal model training...")
        
        model = NodeLevelCausalModel()
        start_time = time.time()
        
        # Train model
        training_result = model.train_model(cfg_files)
        training_time = time.time() - start_time
        
        if training_result['success']:
            # Save model
            model_path = os.path.join(self.models_dir, "causal_node_level.pth")
            save_success = model.save_model(model_path)
            
            # Test model
            test_results = self.test_model(model, cfg_files, "Causal")
            
            result = {
                'model': 'Causal',
                'training_time': training_time,
                'training_result': training_result,
                'model_saved': save_success,
                'model_path': model_path,
                'test_results': test_results
            }
        else:
            result = {
                'model': 'Causal',
                'training_time': training_time,
                'error': training_result.get('error', 'Training failed'),
                'success': False
            }
        
        debug_log(f"Causal training completed: {result}")
        return result
    
    def test_model(self, model, cfg_files: List[Dict], model_name: str) -> Dict:
        """Test model on all CFG files"""
        debug_log(f"Testing {model_name} model...")
        
        total_targets = 0
        total_predictions = 0
        total_time = 0
        
        for cfg_file in cfg_files:
            start_time = time.time()
            
            predictions = model.predict_annotation_targets(cfg_file['data'])
            
            prediction_time = time.time() - start_time
            total_time += prediction_time
            total_predictions += len(predictions)
            
            # Count annotation targets in this CFG
            from node_level_models import NodeClassifier
            nodes = cfg_file['data'].get('nodes', [])
            targets = sum(1 for node in nodes if NodeClassifier.is_annotation_target(node))
            total_targets += targets
            
            debug_log(f"{model_name} test on {cfg_file['method']}: {len(predictions)} predictions, {targets} targets")
        
        test_results = {
            'total_targets': total_targets,
            'total_predictions': total_predictions,
            'total_time': total_time,
            'avg_time_per_cfg': total_time / len(cfg_files) if cfg_files else 0,
            'prediction_rate': total_predictions / total_targets if total_targets > 0 else 0
        }
        
        debug_log(f"{model_name} test results: {test_results}")
        return test_results
    
    def run_complete_training(self) -> Dict:
        """Run complete training pipeline for all models"""
        debug_log("Starting complete training pipeline...")
        
        # Load data
        cfg_files = self.load_cfg_data()
        if not cfg_files:
            return {'error': 'No CFG files found'}
        
        results = {}
        
        # Train all models
        results['HGT'] = self.train_hgt_model(cfg_files)
        results['GBT'] = self.train_gbt_model(cfg_files)
        results['Causal'] = self.train_causal_model(cfg_files)
        
        # Calculate overall statistics
        results['summary'] = self.calculate_summary_statistics(results)
        
        # Save results
        results_path = os.path.join(self.models_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        debug_log(f"Complete training pipeline finished. Results saved to {results_path}")
        return results
    
    def calculate_summary_statistics(self, results: Dict) -> Dict:
        """Calculate summary statistics across all models"""
        summary = {
            'total_models_trained': 0,
            'successful_models': 0,
            'failed_models': 0,
            'total_training_time': 0,
            'average_training_time': 0,
            'model_performance': {}
        }
        
        for model_name in ['HGT', 'GBT', 'Causal']:
            if model_name in results:
                result = results[model_name]
                summary['total_models_trained'] += 1
                
                if result.get('training_result', {}).get('success', False):
                    summary['successful_models'] += 1
                    summary['total_training_time'] += result.get('training_time', 0)
                    
                    # Add performance metrics
                    test_results = result.get('test_results', {})
                    summary['model_performance'][model_name] = {
                        'prediction_rate': test_results.get('prediction_rate', 0),
                        'avg_prediction_time': test_results.get('avg_time_per_cfg', 0),
                        'total_predictions': test_results.get('total_predictions', 0)
                    }
                else:
                    summary['failed_models'] += 1
        
        if summary['successful_models'] > 0:
            summary['average_training_time'] = summary['total_training_time'] / summary['successful_models']
        
        return summary
    
    def print_results_summary(self, results: Dict):
        """Print comprehensive results summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE NODE-LEVEL MODEL TRAINING RESULTS")
        print("="*80)
        
        summary = results.get('summary', {})
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"  Total Models: {summary.get('total_models_trained', 0)}")
        print(f"  Successful: {summary.get('successful_models', 0)}")
        print(f"  Failed: {summary.get('failed_models', 0)}")
        print(f"  Total Training Time: {summary.get('total_training_time', 0):.2f}s")
        print(f"  Average Training Time: {summary.get('average_training_time', 0):.2f}s")
        
        print(f"\n🤖 MODEL DETAILS:")
        for model_name in ['HGT', 'GBT', 'Causal']:
            if model_name in results:
                result = results[model_name]
                print(f"\n{model_name} Model:")
                
                if result.get('training_result', {}).get('success', False):
                    training = result['training_result']
                    test = result.get('test_results', {})
                    
                    print(f"  ✅ Training: SUCCESS")
                    print(f"  📈 Training Accuracy: {training.get('final_accuracy', 0):.3f}")
                    print(f"  ⏱️ Training Time: {result.get('training_time', 0):.2f}s")
                    print(f"  💾 Model Saved: {result.get('model_saved', False)}")
                    print(f"  🎯 Prediction Rate: {test.get('prediction_rate', 0):.3f}")
                    print(f"  📊 Total Predictions: {test.get('total_predictions', 0)}")
                else:
                    print(f"  ❌ Training: FAILED")
                    print(f"  🐛 Error: {result.get('error', 'Unknown')}")
        
        print(f"\n🏆 PERFORMANCE COMPARISON:")
        performance = summary.get('model_performance', {})
        if performance:
            sorted_models = sorted(performance.items(), 
                                 key=lambda x: x[1]['prediction_rate'], 
                                 reverse=True)
            
            for i, (model, metrics) in enumerate(sorted_models, 1):
                print(f"  {i}. {model}: {metrics['prediction_rate']:.3f} prediction rate")
        
        print("\n" + "="*80)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Complete node-level model training pipeline')
    parser.add_argument('--cfg_dir', default='test_results/model_evaluation/cfg_output',
                       help='Directory containing CFG files')
    parser.add_argument('--models_dir', default='models_node_level',
                       help='Directory to save trained models')
    parser.add_argument('--models', nargs='+', default=['HGT', 'GBT', 'Causal'],
                       help='Models to train')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ComprehensiveTrainingPipeline(args.cfg_dir, args.models_dir)
    
    # Run training
    results = pipeline.run_complete_training()
    
    # Print results
    pipeline.print_results_summary(results)

if __name__ == "__main__":
    main()
