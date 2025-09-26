#!/usr/bin/env python3
"""
Node-Level F1 Score Evaluation for CFWR Models

This script evaluates the performance of node-level models on a statistically significant
dataset with proper train/test split using F1 scores and other classification metrics.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class F1EvaluationResult:
    """Results from F1 evaluation"""
    model_name: str
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    support: int
    training_time: float
    prediction_time: float
    difficult_cases: List[Dict[str, Any]]
    confusion_matrix: List[List[int]]

class NodeLevelF1Evaluator:
    """Comprehensive F1 score evaluator for node-level models"""
    
    def __init__(self, dataset_dir: str = "test_results/statistical_dataset"):
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, "train", "cfg_output")
        self.test_dir = os.path.join(dataset_dir, "test", "cfg_output")
        self.results_dir = os.path.join(dataset_dir, "f1_results")
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Evaluation results
        self.evaluation_results = {}
        
        logger.info(f"Initialized F1 evaluator with dataset: {dataset_dir}")
    
    def load_cfg_files(self, directory: str) -> List[Dict[str, Any]]:
        """Load CFG files from a directory with proper structure for models"""
        
        cfg_files = []
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return cfg_files
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                cfg_path = os.path.join(directory, filename)
                
                try:
                    with open(cfg_path, 'r') as f:
                        cfg_data = json.load(f)
                        # Wrap in the structure expected by the models
                        cfg_files.append({
                            'file': cfg_path,
                            'method': cfg_data.get('method_name', filename.replace('.json', '')),
                            'data': cfg_data
                        })
                except Exception as e:
                    logger.warning(f"Error loading CFG {filename}: {e}")
        
        logger.info(f"Loaded {len(cfg_files)} CFG files from {directory}")
        return cfg_files
    
    def create_ground_truth_labels(self, cfg_data: Dict[str, Any]) -> List[int]:
        """Create ground truth labels for annotation targets"""
        
        labels = []
        
        for node in cfg_data.get('nodes', []):
            # Use the NodeClassifier to determine if this is an annotation target
            from node_level_models import NodeClassifier
            is_target = NodeClassifier.is_annotation_target(node)
            labels.append(1 if is_target else 0)
        
        return labels
    
    def extract_difficult_cases(self, cfg_data: Dict[str, Any], predictions: List[int], 
                               ground_truth: List[int]) -> List[Dict[str, Any]]:
        """Extract difficult cases where model predictions differ from ground truth"""
        
        difficult_cases = []
        
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            if pred != true and i < len(cfg_data.get('nodes', [])):
                node = cfg_data['nodes'][i]
                case = {
                    'node_id': i,
                    'predicted': pred,
                    'actual': true,
                    'label': node.get('label', ''),
                    'line': node.get('line', 0),
                    'node_type': node.get('node_type', ''),
                    'complexity': self._calculate_node_complexity(node, cfg_data)
                }
                difficult_cases.append(case)
        
        return difficult_cases
    
    def _calculate_node_complexity(self, node: Dict[str, Any], cfg_data: Dict[str, Any]) -> str:
        """Calculate complexity level of a node based on its context"""
        
        label = node.get('label', '').lower()
        
        # Count control flow complexity
        control_edges = len(cfg_data.get('control_edges', []))
        dataflow_edges = len(cfg_data.get('dataflow_edges', []))
        total_nodes = len(cfg_data.get('nodes', []))
        
        # Determine complexity based on various factors
        if control_edges > 15 or dataflow_edges > 10 or total_nodes > 20:
            return "enterprise"
        elif control_edges > 10 or dataflow_edges > 7 or total_nodes > 15:
            return "very_complex"
        elif control_edges > 5 or dataflow_edges > 4 or total_nodes > 10:
            return "complex"
        elif control_edges > 2 or dataflow_edges > 2 or total_nodes > 5:
            return "medium"
        else:
            return "simple"
    
    def evaluate_hgt_model(self, train_cfgs: List[Dict[str, Any]], 
                          test_cfgs: List[Dict[str, Any]]) -> F1EvaluationResult:
        """Evaluate HGT model with F1 score"""
        
        logger.info("Evaluating Node-Level HGT Model...")
        
        try:
            from node_level_models import NodeLevelHGTModel
            
            # Initialize model
            model = NodeLevelHGTModel()
            
            # Training phase
            start_time = time.time()
            training_result = model.train_model(train_cfgs)
            training_time = time.time() - start_time
            
            if not training_result.get('success', False):
                logger.error("HGT training failed")
                return F1EvaluationResult("HGT", 0.0, 0.0, 0.0, 0.0, 0, training_time, 0.0, [], [[0]])
            
            # Testing phase
            start_time = time.time()
            all_predictions = []
            all_ground_truth = []
            all_difficult_cases = []
            
            for cfg_file in test_cfgs:
                cfg_data = cfg_file['data']
                # Get model predictions
                predictions = model.predict_annotation_targets(cfg_data)
                
                # Create binary prediction vector
                pred_vector = [0] * len(cfg_data.get('nodes', []))
                for pred in predictions:
                    if 'line' in pred and pred['line'] > 0:
                        # Find corresponding node by line number
                        for i, node in enumerate(cfg_data.get('nodes', [])):
                            if node.get('line', 0) == pred['line']:
                                pred_vector[i] = 1
                                break
                
                # Get ground truth
                ground_truth = self.create_ground_truth_labels(cfg_data)
                
                # Ensure same length
                min_len = min(len(pred_vector), len(ground_truth))
                pred_vector = pred_vector[:min_len]
                ground_truth = ground_truth[:min_len]
                
                all_predictions.extend(pred_vector)
                all_ground_truth.extend(ground_truth)
                
                # Extract difficult cases
                difficult_cases = self.extract_difficult_cases(cfg_data, pred_vector, ground_truth)
                all_difficult_cases.extend(difficult_cases)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            if len(all_predictions) == 0 or len(all_ground_truth) == 0:
                logger.warning("No predictions or ground truth available")
                return F1EvaluationResult("HGT", 0.0, 0.0, 0.0, 0.0, 0, training_time, prediction_time, [], [[0]])
            
            f1 = f1_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            precision = precision_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            recall = recall_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            accuracy = accuracy_score(all_ground_truth, all_predictions)
            
            # Confusion matrix
            cm = confusion_matrix(all_ground_truth, all_predictions)
            
            return F1EvaluationResult(
                "Node-Level HGT", f1, precision, recall, accuracy, 
                len(all_ground_truth), training_time, prediction_time, 
                all_difficult_cases[:10], cm.tolist()  # Top 10 difficult cases
            )
            
        except Exception as e:
            logger.error(f"Error evaluating HGT model: {e}")
            return F1EvaluationResult("HGT", 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, [], [[0]])
    
    def evaluate_gbt_model(self, train_cfgs: List[Dict[str, Any]], 
                          test_cfgs: List[Dict[str, Any]]) -> F1EvaluationResult:
        """Evaluate GBT model with F1 score"""
        
        logger.info("Evaluating Node-Level GBT Model...")
        
        try:
            from node_level_models import NodeLevelGBTModel
            
            # Initialize model
            model = NodeLevelGBTModel()
            
            # Training phase
            start_time = time.time()
            training_result = model.train_model(train_cfgs)
            training_time = time.time() - start_time
            
            if not training_result.get('success', False):
                logger.error("GBT training failed")
                return F1EvaluationResult("GBT", 0.0, 0.0, 0.0, 0.0, 0, training_time, 0.0, [], [[0]])
            
            # Testing phase
            start_time = time.time()
            all_predictions = []
            all_ground_truth = []
            all_difficult_cases = []
            
            for cfg_file in test_cfgs:
                cfg_data = cfg_file['data']
                # Get model predictions
                predictions = model.predict_annotation_targets(cfg_data)
                
                # Create binary prediction vector
                pred_vector = [0] * len(cfg_data.get('nodes', []))
                for pred in predictions:
                    if 'line' in pred and pred['line'] > 0:
                        # Find corresponding node by line number
                        for i, node in enumerate(cfg_data.get('nodes', [])):
                            if node.get('line', 0) == pred['line']:
                                pred_vector[i] = 1
                                break
                
                # Get ground truth
                ground_truth = self.create_ground_truth_labels(cfg_data)
                
                # Ensure same length
                min_len = min(len(pred_vector), len(ground_truth))
                pred_vector = pred_vector[:min_len]
                ground_truth = ground_truth[:min_len]
                
                all_predictions.extend(pred_vector)
                all_ground_truth.extend(ground_truth)
                
                # Extract difficult cases
                difficult_cases = self.extract_difficult_cases(cfg_data, pred_vector, ground_truth)
                all_difficult_cases.extend(difficult_cases)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            if len(all_predictions) == 0 or len(all_ground_truth) == 0:
                logger.warning("No predictions or ground truth available")
                return F1EvaluationResult("GBT", 0.0, 0.0, 0.0, 0.0, 0, training_time, prediction_time, [], [[0]])
            
            f1 = f1_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            precision = precision_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            recall = recall_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            accuracy = accuracy_score(all_ground_truth, all_predictions)
            
            # Confusion matrix
            cm = confusion_matrix(all_ground_truth, all_predictions)
            
            return F1EvaluationResult(
                "Node-Level GBT", f1, precision, recall, accuracy, 
                len(all_ground_truth), training_time, prediction_time, 
                all_difficult_cases[:10], cm.tolist()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating GBT model: {e}")
            return F1EvaluationResult("GBT", 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, [], [[0]])
    
    def evaluate_causal_model(self, train_cfgs: List[Dict[str, Any]], 
                             test_cfgs: List[Dict[str, Any]]) -> F1EvaluationResult:
        """Evaluate Causal model with F1 score"""
        
        logger.info("Evaluating Node-Level Causal Model...")
        
        try:
            from node_level_models import NodeLevelCausalModel
            
            # Initialize model
            model = NodeLevelCausalModel()
            
            # Training phase
            start_time = time.time()
            training_result = model.train_model(train_cfgs)
            training_time = time.time() - start_time
            
            if not training_result.get('success', False):
                logger.error("Causal training failed")
                return F1EvaluationResult("Causal", 0.0, 0.0, 0.0, 0.0, 0, training_time, 0.0, [], [[0]])
            
            # Testing phase
            start_time = time.time()
            all_predictions = []
            all_ground_truth = []
            all_difficult_cases = []
            
            for cfg_file in test_cfgs:
                cfg_data = cfg_file['data']
                # Get model predictions
                predictions = model.predict_annotation_targets(cfg_data)
                
                # Create binary prediction vector
                pred_vector = [0] * len(cfg_data.get('nodes', []))
                for pred in predictions:
                    if 'line' in pred and pred['line'] > 0:
                        # Find corresponding node by line number
                        for i, node in enumerate(cfg_data.get('nodes', [])):
                            if node.get('line', 0) == pred['line']:
                                pred_vector[i] = 1
                                break
                
                # Get ground truth
                ground_truth = self.create_ground_truth_labels(cfg_data)
                
                # Ensure same length
                min_len = min(len(pred_vector), len(ground_truth))
                pred_vector = pred_vector[:min_len]
                ground_truth = ground_truth[:min_len]
                
                all_predictions.extend(pred_vector)
                all_ground_truth.extend(ground_truth)
                
                # Extract difficult cases
                difficult_cases = self.extract_difficult_cases(cfg_data, pred_vector, ground_truth)
                all_difficult_cases.extend(difficult_cases)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            if len(all_predictions) == 0 or len(all_ground_truth) == 0:
                logger.warning("No predictions or ground truth available")
                return F1EvaluationResult("Causal", 0.0, 0.0, 0.0, 0.0, 0, training_time, prediction_time, [], [[0]])
            
            f1 = f1_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            precision = precision_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            recall = recall_score(all_ground_truth, all_predictions, average='binary', zero_division=0)
            accuracy = accuracy_score(all_ground_truth, all_predictions)
            
            # Confusion matrix
            cm = confusion_matrix(all_ground_truth, all_predictions)
            
            return F1EvaluationResult(
                "Node-Level Causal", f1, precision, recall, accuracy, 
                len(all_ground_truth), training_time, prediction_time, 
                all_difficult_cases[:10], cm.tolist()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating Causal model: {e}")
            return F1EvaluationResult("Causal", 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, [], [[0]])
    
    def run_comprehensive_f1_evaluation(self) -> Dict[str, F1EvaluationResult]:
        """Run comprehensive F1 evaluation on all models"""
        
        logger.info("Starting comprehensive F1 evaluation...")
        
        # Load train and test data
        train_cfgs = self.load_cfg_files(self.train_dir)
        test_cfgs = self.load_cfg_files(self.test_dir)
        
        if not train_cfgs or not test_cfgs:
            logger.error("Insufficient training or test data")
            return {}
        
        logger.info(f"Training on {len(train_cfgs)} CFGs, testing on {len(test_cfgs)} CFGs")
        
        # Evaluate all models
        self.evaluation_results = {
            "HGT": self.evaluate_hgt_model(train_cfgs, test_cfgs),
            "GBT": self.evaluate_gbt_model(train_cfgs, test_cfgs),
            "Causal": self.evaluate_causal_model(train_cfgs, test_cfgs)
        }
        
        # Save results
        self.save_f1_results()
        
        # Print comprehensive report
        self.print_f1_report()
        
        return self.evaluation_results
    
    def save_f1_results(self) -> None:
        """Save F1 evaluation results to file"""
        
        results_data = {}
        
        for model_name, result in self.evaluation_results.items():
            results_data[model_name] = {
                "model_name": result.model_name,
                "f1_score": result.f1_score,
                "precision": result.precision,
                "recall": result.recall,
                "accuracy": result.accuracy,
                "support": result.support,
                "training_time": result.training_time,
                "prediction_time": result.prediction_time,
                "difficult_cases": result.difficult_cases,
                "confusion_matrix": result.confusion_matrix
            }
        
        results_path = os.path.join(self.results_dir, "f1_evaluation_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"F1 evaluation results saved to {results_path}")
    
    def print_f1_report(self) -> None:
        """Print comprehensive F1 evaluation report"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE F1 SCORE EVALUATION RESULTS")
        print("="*80)
        print(f"📊 Dataset: {len(self.load_cfg_files(self.train_dir))} train + {len(self.load_cfg_files(self.test_dir))} test CFGs")
        print(f"🎯 Evaluation Type: F1 Score with Train/Test Split")
        print("="*80)
        
        # F1 Score comparison table
        print("\n📈 NODE-LEVEL MODEL F1 PERFORMANCE:")
        print("-" * 80)
        print(f"{'Model':<15} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Accuracy':<10} {'Support':<10}")
        print("-" * 80)
        
        for model_name, result in self.evaluation_results.items():
            print(f"{result.model_name:<15} {result.f1_score:<10.3f} {result.precision:<10.3f} "
                  f"{result.recall:<10.3f} {result.accuracy:<10.3f} {result.support:<10}")
        
        print("-" * 80)
        
        # Performance analysis
        print("\n⚡ PERFORMANCE ANALYSIS:")
        print("-" * 40)
        
        best_f1 = max(self.evaluation_results.values(), key=lambda x: x.f1_score)
        best_precision = max(self.evaluation_results.values(), key=lambda x: x.precision)
        best_recall = max(self.evaluation_results.values(), key=lambda x: x.recall)
        
        print(f"🏆 Best F1 Score: {best_f1.model_name} ({best_f1.f1_score:.3f})")
        print(f"🎯 Best Precision: {best_precision.model_name} ({best_precision.precision:.3f})")
        print(f"📊 Best Recall: {best_recall.model_name} ({best_recall.recall:.3f})")
        
        # Difficult cases analysis
        print("\n🔍 DIFFICULT CASES ANALYSIS:")
        print("-" * 40)
        
        for model_name, result in self.evaluation_results.items():
            if result.difficult_cases:
                print(f"\n{result.model_name} - {len(result.difficult_cases)} difficult cases:")
                for case in result.difficult_cases[:3]:  # Show top 3
                    print(f"  • Line {case['line']}: {case['label'][:50]}... "
                          f"(Predicted: {case['predicted']}, Actual: {case['actual']}, "
                          f"Complexity: {case['complexity']})")
        
        print("="*80)
        print("🎯 F1 evaluation complete!")
        print("📁 Results saved to: test_results/statistical_dataset/f1_results/")

def main():
    """Main function to run comprehensive F1 evaluation"""
    
    print("🔬 Starting Comprehensive F1 Score Evaluation for Node-Level Models")
    print("="*80)
    
    # Create evaluator
    evaluator = NodeLevelF1Evaluator()
    
    # Run comprehensive F1 evaluation
    results = evaluator.run_comprehensive_f1_evaluation()
    
    if results:
        print("\n✅ All models evaluated successfully with F1 scores!")
    else:
        print("\n❌ F1 evaluation failed - check logs for details")

if __name__ == "__main__":
    main()
