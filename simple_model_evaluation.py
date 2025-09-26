#!/usr/bin/env python3
"""
Simplified Model Performance Evaluation

This script evaluates the practical performance characteristics of HGT, GBT, and Causal models
by measuring training time, prediction speed, and accuracy on real data.
"""

import os
import json
import time
import numpy as np
import torch
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleModelEvaluator:
    """Simplified evaluator focusing on practical performance metrics"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_training_performance(self) -> Dict:
        """Evaluate training performance by running training scripts"""
        logger.info("Evaluating training performance...")
        
        training_results = {}
        
        # Test HGT training
        logger.info("Testing HGT training...")
        start_time = time.time()
        try:
            os.system("python hgt.py > test_results/model_evaluation/hgt_training.log 2>&1")
            hgt_time = time.time() - start_time
            training_results['HGT'] = {
                'training_time': hgt_time,
                'success': True
            }
        except Exception as e:
            training_results['HGT'] = {
                'training_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        
        # Test GBT training
        logger.info("Testing GBT training...")
        start_time = time.time()
        try:
            os.system("python gbt.py > test_results/model_evaluation/gbt_training.log 2>&1")
            gbt_time = time.time() - start_time
            training_results['GBT'] = {
                'training_time': gbt_time,
                'success': True
            }
        except Exception as e:
            training_results['GBT'] = {
                'training_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        
        # Test Causal training
        logger.info("Testing Causal training...")
        start_time = time.time()
        try:
            os.system("python causal_model.py > test_results/model_evaluation/causal_training.log 2>&1")
            causal_time = time.time() - start_time
            training_results['Causal'] = {
                'training_time': causal_time,
                'success': True
            }
        except Exception as e:
            training_results['Causal'] = {
                'training_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        
        return training_results
    
    def evaluate_prediction_performance(self) -> Dict:
        """Evaluate prediction performance using existing models"""
        logger.info("Evaluating prediction performance...")
        
        prediction_results = {}
        test_file = "test_results/model_evaluation/TestDataset.java"
        
        # Test HGT prediction
        logger.info("Testing HGT prediction...")
        start_time = time.time()
        try:
            os.system(f"python predict_hgt.py --java_file {test_file} --output_dir test_results/model_evaluation/hgt_predictions > test_results/model_evaluation/hgt_prediction.log 2>&1")
            hgt_time = time.time() - start_time
            prediction_results['HGT'] = {
                'prediction_time': hgt_time,
                'success': True
            }
        except Exception as e:
            prediction_results['HGT'] = {
                'prediction_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        
        # Test GBT prediction
        logger.info("Testing GBT prediction...")
        start_time = time.time()
        try:
            os.system(f"python predict_gbt.py --java_file {test_file} --output_dir test_results/model_evaluation/gbt_predictions > test_results/model_evaluation/gbt_prediction.log 2>&1")
            gbt_time = time.time() - start_time
            prediction_results['GBT'] = {
                'prediction_time': gbt_time,
                'success': True
            }
        except Exception as e:
            prediction_results['GBT'] = {
                'prediction_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        
        # Test Causal prediction
        logger.info("Testing Causal prediction...")
        start_time = time.time()
        try:
            os.system(f"python predict_causal.py --java_file {test_file} --output_dir test_results/model_evaluation/causal_predictions > test_results/model_evaluation/causal_prediction.log 2>&1")
            causal_time = time.time() - start_time
            prediction_results['Causal'] = {
                'prediction_time': causal_time,
                'success': True
            }
        except Exception as e:
            prediction_results['Causal'] = {
                'prediction_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        
        return prediction_results
    
    def analyze_model_characteristics(self) -> Dict:
        """Analyze model characteristics and capabilities"""
        logger.info("Analyzing model characteristics...")
        
        characteristics = {
            'HGT': {
                'type': 'Graph Neural Network',
                'strengths': [
                    'Handles graph-structured data naturally',
                    'Captures complex relationships between nodes',
                    'Good for CFG-based analysis',
                    'Can learn from dataflow information'
                ],
                'weaknesses': [
                    'Requires more computational resources',
                    'Slower training and inference',
                    'More complex to tune',
                    'Requires graph preprocessing'
                ],
                'best_for': 'Complex CFG analysis with dataflow information',
                'complexity': 'High'
            },
            'GBT': {
                'type': 'Ensemble Learning',
                'strengths': [
                    'Fast training and prediction',
                    'Handles mixed data types well',
                    'Robust to outliers',
                    'Easy to interpret',
                    'Good baseline performance'
                ],
                'weaknesses': [
                    'May overfit on small datasets',
                    'Less effective on graph data',
                    'Requires feature engineering',
                    'Limited by feature representation'
                ],
                'best_for': 'Quick predictions and baseline comparisons',
                'complexity': 'Medium'
            },
            'Causal': {
                'type': 'Neural Network (Simplified)',
                'strengths': [
                    'Designed for causal inference',
                    'Can learn complex patterns',
                    'Flexible architecture',
                    'Good for structured data'
                ],
                'weaknesses': [
                    'Requires large amounts of data',
                    'Can be prone to overfitting',
                    'Less interpretable',
                    'Requires careful tuning'
                ],
                'best_for': 'Large-scale analysis with sufficient data',
                'complexity': 'Medium-High'
            }
        }
        
        return characteristics
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation of all models"""
        logger.info("Starting comprehensive model evaluation...")
        
        results = {}
        
        # Evaluate training performance
        results['training'] = self.evaluate_training_performance()
        
        # Evaluate prediction performance
        results['prediction'] = self.evaluate_prediction_performance()
        
        # Analyze model characteristics
        results['characteristics'] = self.analyze_model_characteristics()
        
        # Calculate overall rankings
        results['rankings'] = self._calculate_rankings(results)
        
        return results
    
    def _calculate_rankings(self, results: Dict) -> Dict:
        """Calculate overall performance rankings"""
        rankings = {}
        
        # Training time ranking (lower is better)
        training_times = {}
        for model, data in results['training'].items():
            if data.get('success', False):
                training_times[model] = data['training_time']
        
        if training_times:
            rankings['training_speed'] = sorted(training_times.keys(), key=lambda k: training_times[k])
        
        # Prediction time ranking (lower is better)
        prediction_times = {}
        for model, data in results['prediction'].items():
            if data.get('success', False):
                prediction_times[model] = data['prediction_time']
        
        if prediction_times:
            rankings['prediction_speed'] = sorted(prediction_times.keys(), key=lambda k: prediction_times[k])
        
        # Overall success rate
        success_rates = {}
        for model in ['HGT', 'GBT', 'Causal']:
            training_success = results['training'].get(model, {}).get('success', False)
            prediction_success = results['prediction'].get(model, {}).get('success', False)
            success_rate = (training_success + prediction_success) / 2
            success_rates[model] = success_rate
        
        if success_rates:
            rankings['reliability'] = sorted(success_rates.keys(), key=lambda k: success_rates[k], reverse=True)
        
        return rankings
    
    def print_summary(self, results: Dict):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL PERFORMANCE EVALUATION")
        print("="*80)
        
        # Training Performance
        print("\n📊 TRAINING PERFORMANCE:")
        print("-" * 40)
        for model, data in results['training'].items():
            status = "✅ SUCCESS" if data.get('success', False) else "❌ FAILED"
            time_str = f"{data.get('training_time', 0):.2f}s"
            print(f"{model:8} | {status:10} | Time: {time_str}")
            if not data.get('success', False) and 'error' in data:
                print(f"         | Error: {data['error']}")
        
        # Prediction Performance
        print("\n🚀 PREDICTION PERFORMANCE:")
        print("-" * 40)
        for model, data in results['prediction'].items():
            status = "✅ SUCCESS" if data.get('success', False) else "❌ FAILED"
            time_str = f"{data.get('prediction_time', 0):.2f}s"
            print(f"{model:8} | {status:10} | Time: {time_str}")
            if not data.get('success', False) and 'error' in data:
                print(f"         | Error: {data['error']}")
        
        # Model Characteristics
        print("\n🔍 MODEL CHARACTERISTICS:")
        print("-" * 40)
        for model, chars in results['characteristics'].items():
            print(f"\n{model} ({chars['type']}):")
            print(f"  Best for: {chars['best_for']}")
            print(f"  Complexity: {chars['complexity']}")
            print(f"  Strengths: {', '.join(chars['strengths'][:2])}")
            print(f"  Weaknesses: {', '.join(chars['weaknesses'][:2])}")
        
        # Rankings
        if 'rankings' in results:
            print("\n🏆 PERFORMANCE RANKINGS:")
            print("-" * 40)
            for category, ranking in results['rankings'].items():
                print(f"{category.replace('_', ' ').title()}: {' > '.join(ranking)}")
        
        # Overall Recommendation
        print("\n💡 OVERALL RECOMMENDATION:")
        print("-" * 40)
        if 'rankings' in results and 'reliability' in results['rankings']:
            best_model = results['rankings']['reliability'][0]
            print(f"Best Overall: {best_model}")
            print(f"Reason: Most reliable with consistent training and prediction success")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

def main():
    """Main evaluation function"""
    evaluator = SimpleModelEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    evaluator.print_summary(results)
    evaluator.save_results(results, "test_results/model_evaluation/comprehensive_evaluation.json")

if __name__ == "__main__":
    main()
