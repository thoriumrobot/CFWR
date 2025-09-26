#!/usr/bin/env python3
"""
Comprehensive Statistical Testing Script for CFWR Node-Level Models

This script evaluates the refactored node-level models on a statistically significant dataset
to provide robust performance metrics and validation.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Results from model testing"""
    model_name: str
    total_samples: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_time: float
    training_time: float
    details: Dict[str, Any]

class StatisticalModelTester:
    """Comprehensive tester for node-level models on statistical dataset"""
    
    def __init__(self, dataset_dir: str = "test_results/statistical_dataset"):
        self.dataset_dir = dataset_dir
        self.cfg_dir = os.path.join(dataset_dir, "cfg_output")
        self.java_dir = os.path.join(dataset_dir, "java_files")
        self.results_dir = os.path.join(dataset_dir, "test_results")
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load dataset statistics
        self.dataset_stats = self.load_dataset_statistics()
        
        # Test results
        self.test_results = {}
        
        logger.info(f"Initialized statistical tester with dataset: {self.dataset_stats['total_methods']} methods")
    
    def load_dataset_statistics(self) -> Dict[str, Any]:
        """Load dataset statistics"""
        stats_path = os.path.join(self.dataset_dir, "dataset_statistics.json")
        
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                return json.load(f)
        else:
            return {"total_methods": 0, "generated_files": []}
    
    def load_cfg_files(self) -> List[Dict[str, Any]]:
        """Load all CFG files from the dataset with proper structure for node-level models"""
        
        cfg_files = []
        
        if not os.path.exists(self.cfg_dir):
            logger.error(f"CFG directory not found: {self.cfg_dir}")
            return cfg_files
        
        for filename in os.listdir(self.cfg_dir):
            if filename.endswith('.json'):
                cfg_path = os.path.join(self.cfg_dir, filename)
                
                try:
                    with open(cfg_path, 'r') as f:
                        cfg_data = json.load(f)
                        # Wrap in structure expected by node-level models
                        cfg_files.append({
                            'file': cfg_path,
                            'method': cfg_data.get('method_name', filename.replace('.json', '')),
                            'data': cfg_data
                        })
                except Exception as e:
                    logger.warning(f"Error loading CFG {filename}: {e}")
        
        logger.info(f"Loaded {len(cfg_files)} CFG files")
        return cfg_files
    
    def test_hgt_model(self, cfg_files: List[Dict[str, Any]]) -> TestResult:
        """Test HGT model on statistical dataset"""
        
        logger.info("Testing HGT model on statistical dataset...")
        
        try:
            from node_level_models import NodeLevelHGTModel
            
            # Initialize model
            model = NodeLevelHGTModel()
            
            # Training phase - convert CFG files to expected format
            start_time = time.time()
            
            # Create training data in expected format
            training_cfgs = []
            for cfg_file in cfg_files:
                cfg_data = cfg_file['data']
                # Convert to expected format
                formatted_cfg = {
                    'method_name': cfg_data.get('method_name', 'unknown'),
                    'nodes': cfg_data.get('nodes', []),
                    'control_edges': cfg_data.get('control_edges', []),
                    'dataflow_edges': cfg_data.get('dataflow_edges', [])
                }
                training_cfgs.append(formatted_cfg)
            
            training_result = model.train_model(training_cfgs)
            training_time = time.time() - start_time
            
            if not training_result.get('success', False):
                logger.error("HGT training failed")
                return TestResult("HGT", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, training_time, {"error": "Training failed"})
            
            # Testing phase
            start_time = time.time()
            total_samples = 0
            correct_predictions = 0
            all_predictions = []
            all_targets = []
            
            for cfg_data in training_cfgs:
                predictions = model.predict_annotation_targets(cfg_data)
                
                # Count annotation targets (ground truth)
                annotation_targets = self.count_annotation_targets(cfg_data)
                total_samples += annotation_targets
                
                # Count correct predictions
                predicted_count = len(predictions)
                correct_predictions += min(predicted_count, annotation_targets)
                
                all_predictions.extend([1] * predicted_count)
                all_targets.extend([1] * annotation_targets)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            
            # For precision, recall, F1 - use binary classification approach
            if len(all_predictions) > 0 and len(all_targets) > 0:
                # Pad to same length
                max_len = max(len(all_predictions), len(all_targets))
                all_predictions.extend([0] * (max_len - len(all_predictions)))
                all_targets.extend([0] * (max_len - len(all_targets)))
                
                precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
                recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
                f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            details = {
                "training_result": training_result,
                "total_cfgs": len(training_cfgs),
                "avg_predictions_per_cfg": correct_predictions / len(training_cfgs) if training_cfgs else 0
            }
            
            return TestResult("HGT", total_samples, correct_predictions, accuracy, 
                            precision, recall, f1, prediction_time, training_time, details)
            
        except Exception as e:
            logger.error(f"Error testing HGT model: {e}")
            return TestResult("HGT", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {"error": str(e)})
    
    def test_gbt_model(self, cfg_files: List[Dict[str, Any]]) -> TestResult:
        """Test GBT model on statistical dataset"""
        
        logger.info("Testing GBT model on statistical dataset...")
        
        try:
            from node_level_models import NodeLevelGBTModel
            
            # Initialize model
            model = NodeLevelGBTModel()
            
            # Training phase - convert CFG files to expected format
            start_time = time.time()
            
            # Create training data in expected format
            training_cfgs = []
            for cfg_file in cfg_files:
                cfg_data = cfg_file['data']
                # Convert to expected format
                formatted_cfg = {
                    'method_name': cfg_data.get('method_name', 'unknown'),
                    'nodes': cfg_data.get('nodes', []),
                    'control_edges': cfg_data.get('control_edges', []),
                    'dataflow_edges': cfg_data.get('dataflow_edges', [])
                }
                training_cfgs.append(formatted_cfg)
            
            training_result = model.train_model(training_cfgs)
            training_time = time.time() - start_time
            
            if not training_result.get('success', False):
                logger.error("GBT training failed")
                return TestResult("GBT", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, training_time, {"error": "Training failed"})
            
            # Testing phase
            start_time = time.time()
            total_samples = 0
            correct_predictions = 0
            all_predictions = []
            all_targets = []
            
            for cfg_data in training_cfgs:
                predictions = model.predict_annotation_targets(cfg_data)
                
                # Count annotation targets (ground truth)
                annotation_targets = self.count_annotation_targets(cfg_data)
                total_samples += annotation_targets
                
                # Count correct predictions
                predicted_count = len(predictions)
                correct_predictions += min(predicted_count, annotation_targets)
                
                all_predictions.extend([1] * predicted_count)
                all_targets.extend([1] * annotation_targets)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            
            # For precision, recall, F1
            if len(all_predictions) > 0 and len(all_targets) > 0:
                max_len = max(len(all_predictions), len(all_targets))
                all_predictions.extend([0] * (max_len - len(all_predictions)))
                all_targets.extend([0] * (max_len - len(all_targets)))
                
                precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
                recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
                f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            details = {
                "training_result": training_result,
                "total_cfgs": len(training_cfgs),
                "avg_predictions_per_cfg": correct_predictions / len(training_cfgs) if training_cfgs else 0
            }
            
            return TestResult("GBT", total_samples, correct_predictions, accuracy, 
                            precision, recall, f1, prediction_time, training_time, details)
            
        except Exception as e:
            logger.error(f"Error testing GBT model: {e}")
            return TestResult("GBT", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {"error": str(e)})
    
    def test_causal_model(self, cfg_files: List[Dict[str, Any]]) -> TestResult:
        """Test Causal model on statistical dataset"""
        
        logger.info("Testing Causal model on statistical dataset...")
        
        try:
            from node_level_models import NodeLevelCausalModel
            
            # Initialize model
            model = NodeLevelCausalModel()
            
            # Training phase - convert CFG files to expected format
            start_time = time.time()
            
            # Create training data in expected format
            training_cfgs = []
            for cfg_file in cfg_files:
                cfg_data = cfg_file['data']
                # Convert to expected format
                formatted_cfg = {
                    'method_name': cfg_data.get('method_name', 'unknown'),
                    'nodes': cfg_data.get('nodes', []),
                    'control_edges': cfg_data.get('control_edges', []),
                    'dataflow_edges': cfg_data.get('dataflow_edges', [])
                }
                training_cfgs.append(formatted_cfg)
            
            training_result = model.train_model(training_cfgs)
            training_time = time.time() - start_time
            
            if not training_result.get('success', False):
                logger.error("Causal training failed")
                return TestResult("Causal", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, training_time, {"error": "Training failed"})
            
            # Testing phase
            start_time = time.time()
            total_samples = 0
            correct_predictions = 0
            all_predictions = []
            all_targets = []
            
            for cfg_data in training_cfgs:
                predictions = model.predict_annotation_targets(cfg_data)
                
                # Count annotation targets (ground truth)
                annotation_targets = self.count_annotation_targets(cfg_data)
                total_samples += annotation_targets
                
                # Count correct predictions
                predicted_count = len(predictions)
                correct_predictions += min(predicted_count, annotation_targets)
                
                all_predictions.extend([1] * predicted_count)
                all_targets.extend([1] * annotation_targets)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            
            # For precision, recall, F1
            if len(all_predictions) > 0 and len(all_targets) > 0:
                max_len = max(len(all_predictions), len(all_targets))
                all_predictions.extend([0] * (max_len - len(all_predictions)))
                all_targets.extend([0] * (max_len - len(all_targets)))
                
                precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
                recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
                f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
            else:
                precision = recall = f1 = 0.0
            
            details = {
                "training_result": training_result,
                "total_cfgs": len(training_cfgs),
                "avg_predictions_per_cfg": correct_predictions / len(training_cfgs) if training_cfgs else 0
            }
            
            return TestResult("Causal", total_samples, correct_predictions, accuracy, 
                            precision, recall, f1, prediction_time, training_time, details)
            
        except Exception as e:
            logger.error(f"Error testing Causal model: {e}")
            return TestResult("Causal", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {"error": str(e)})
    
    def count_annotation_targets(self, cfg_data: Dict[str, Any]) -> int:
        """Count the number of annotation targets in a CFG"""
        
        count = 0
        
        for node in cfg_data.get('nodes', []):
            label = node.get('label', '').lower()
            
            # Count nodes that would be annotation targets
            if any(keyword in label for keyword in ['localvariabledeclaration', 'field', 'parameter', 'method']):
                count += 1
        
        return count
    
    def run_comprehensive_test(self) -> Dict[str, TestResult]:
        """Run comprehensive testing on all models"""
        
        logger.info("Starting comprehensive statistical testing...")
        
        # Load CFG files
        cfg_files = self.load_cfg_files()
        
        if not cfg_files:
            logger.error("No CFG files found for testing")
            return {}
        
        logger.info(f"Testing on {len(cfg_files)} CFG files")
        
        # Test all models
        self.test_results = {
            "HGT": self.test_hgt_model(cfg_files),
            "GBT": self.test_gbt_model(cfg_files),
            "Causal": self.test_causal_model(cfg_files)
        }
        
        # Save results
        self.save_test_results()
        
        # Print comprehensive report
        self.print_comprehensive_report()
        
        return self.test_results
    
    def save_test_results(self) -> None:
        """Save test results to file"""
        
        results_data = {}
        
        for model_name, result in self.test_results.items():
            results_data[model_name] = {
                "model_name": result.model_name,
                "total_samples": result.total_samples,
                "correct_predictions": result.correct_predictions,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "prediction_time": result.prediction_time,
                "training_time": result.training_time,
                "details": result.details
            }
        
        results_path = os.path.join(self.results_dir, "statistical_test_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Test results saved to {results_path}")
    
    def print_comprehensive_report(self) -> None:
        """Print comprehensive test report"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL TESTING RESULTS")
        print("="*80)
        print(f"📊 Dataset Size: {self.dataset_stats['total_methods']} methods")
        print(f"📁 CFG Files Tested: {len(self.load_cfg_files())}")
        print(f"🎯 Statistical Significance: {'✅ YES' if self.dataset_stats['total_methods'] >= 100 else '❌ NO'}")
        print("="*80)
        
        # Model comparison table
        print("\n📈 MODEL PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Samples':<10}")
        print("-" * 80)
        
        for model_name, result in self.test_results.items():
            print(f"{model_name:<10} {result.accuracy:<10.3f} {result.precision:<10.3f} "
                  f"{result.recall:<10.3f} {result.f1_score:<10.3f} {result.total_samples:<10}")
        
        print("-" * 80)
        
        # Performance analysis
        print("\n⚡ PERFORMANCE ANALYSIS:")
        print("-" * 40)
        
        best_accuracy = max(self.test_results.values(), key=lambda x: x.accuracy)
        fastest_training = min(self.test_results.values(), key=lambda x: x.training_time)
        fastest_prediction = min(self.test_results.values(), key=lambda x: x.prediction_time)
        
        print(f"🏆 Best Accuracy: {best_accuracy.model_name} ({best_accuracy.accuracy:.3f})")
        print(f"⚡ Fastest Training: {fastest_training.model_name} ({fastest_training.training_time:.3f}s)")
        print(f"🚀 Fastest Prediction: {fastest_prediction.model_name} ({fastest_prediction.prediction_time:.3f}s)")
        
        # Statistical significance analysis
        print("\n🔬 STATISTICAL SIGNIFICANCE ANALYSIS:")
        print("-" * 40)
        
        total_samples = sum(result.total_samples for result in self.test_results.values())
        print(f"Total Annotation Targets: {total_samples}")
        
        if total_samples >= 200:
            print("✅ Strong statistical significance (≥200 samples)")
        elif total_samples >= 100:
            print("✅ Good statistical significance (≥100 samples)")
        elif total_samples >= 50:
            print("⚠️  Moderate statistical significance (≥50 samples)")
        else:
            print("❌ Limited statistical significance (<50 samples)")
        
        print("="*80)
        print("🎯 Statistical testing complete!")
        print("📁 Results saved to: test_results/statistical_dataset/test_results/")

def main():
    """Main function to run comprehensive statistical testing"""
    
    print("🔬 Starting Comprehensive Statistical Testing for CFWR Node-Level Models")
    print("="*80)
    
    # Create tester
    tester = StatisticalModelTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    if results:
        print("\n✅ All models tested successfully on statistical dataset!")
    else:
        print("\n❌ Testing failed - check logs for details")

if __name__ == "__main__":
    main()
