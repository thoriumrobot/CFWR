#!/usr/bin/env python3
"""
Integration Script for Node-Level Models

This script demonstrates how to integrate the refactored node-level models
into the existing CFWR pipeline, ensuring annotations are only placed
before methods, fields, and parameters.
"""

import os
import json
import argparse
from typing import Dict, List
import logging
from node_level_models import NodeLevelHGTModel, NodeLevelGBTModel, NodeLevelCausalModel, NodeClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NodeLevelPipeline:
    """
    Integrated pipeline using node-level models with annotation target filtering.
    """
    
    def __init__(self):
        self.hgt_model = NodeLevelHGTModel()
        self.gbt_model = NodeLevelGBTModel()
        self.causal_model = NodeLevelCausalModel()
        
    def train_models(self, cfg_files: List[Dict]) -> Dict:
        """
        Train all models on node-level data.
        """
        logger.info("Training node-level models...")
        
        results = {}
        
        # Train GBT
        logger.info("Training node-level GBT...")
        gbt_result = self.gbt_model.train_on_cfgs(cfg_files)
        results['GBT'] = gbt_result
        
        # Train Causal
        logger.info("Training node-level Causal...")
        causal_result = self.causal_model.train_on_cfgs(cfg_files)
        results['Causal'] = causal_result
        
        # HGT doesn't need explicit training for this demo
        results['HGT'] = {'success': True, 'note': 'Using pre-initialized weights'}
        
        return results
    
    def predict_annotation_targets(self, cfg_data: Dict, models: List[str] = None) -> Dict:
        """
        Predict annotation targets using specified models.
        """
        if models is None:
            models = ['HGT', 'GBT', 'Causal']
        
        results = {}
        
        for model_name in models:
            logger.info(f"Predicting with {model_name}...")
            
            if model_name == 'HGT':
                targets = self.hgt_model.predict_annotation_targets(cfg_data)
            elif model_name == 'GBT' and self.gbt_model.is_trained:
                targets = self.gbt_model.predict_annotation_targets(cfg_data)
            elif model_name == 'Causal' and self.causal_model.is_trained:
                targets = self.causal_model.predict_annotation_targets(cfg_data)
            else:
                targets = []
                logger.warning(f"{model_name} model not available or not trained")
            
            results[model_name] = targets
        
        return results
    
    def analyze_annotation_targets(self, cfg_data: Dict) -> Dict:
        """
        Analyze all potential annotation targets in a CFG.
        """
        nodes = cfg_data.get('nodes', [])
        analysis = {
            'total_nodes': len(nodes),
            'annotation_targets': [],
            'target_types': {'method': 0, 'field': 0, 'parameter': 0, 'variable': 0},
            'non_targets': 0
        }
        
        for node in nodes:
            if NodeClassifier.is_annotation_target(node):
                context = NodeClassifier.extract_annotation_context(node)
                analysis['annotation_targets'].append(context)
                
                annotation_type = context['annotation_type']
                if annotation_type in analysis['target_types']:
                    analysis['target_types'][annotation_type] += 1
            else:
                analysis['non_targets'] += 1
        
        return analysis
    
    def generate_placement_report(self, predictions: Dict, cfg_data: Dict) -> Dict:
        """
        Generate a comprehensive report for annotation placement.
        """
        report = {
            'method_name': cfg_data.get('method_name', 'unknown'),
            'total_predictions': 0,
            'model_consensus': [],
            'placement_recommendations': []
        }
        
        # Collect all unique lines from all models
        all_lines = set()
        for model_name, targets in predictions.items():
            for target in targets:
                if target.get('line'):
                    all_lines.add(target['line'])
            report['total_predictions'] += len(targets)
        
        # Analyze consensus
        for line in all_lines:
            consensus = {
                'line': line,
                'models_agreeing': [],
                'annotation_contexts': [],
                'confidence_scores': []
            }
            
            for model_name, targets in predictions.items():
                for target in targets:
                    if target.get('line') == line:
                        consensus['models_agreeing'].append(model_name)
                        consensus['annotation_contexts'].append(target)
                        consensus['confidence_scores'].append(target.get('prediction_score', 0))
            
            if len(consensus['models_agreeing']) >= 2:  # At least 2 models agree
                report['model_consensus'].append(consensus)
            
            # Generate placement recommendation
            if consensus['confidence_scores']:
                avg_confidence = sum(consensus['confidence_scores']) / len(consensus['confidence_scores'])
                recommendation = {
                    'line': line,
                    'models_count': len(consensus['models_agreeing']),
                    'average_confidence': avg_confidence,
                    'annotation_type': consensus['annotation_contexts'][0].get('annotation_type', 'unknown'),
                    'recommendation': 'place' if avg_confidence > 0.5 else 'consider'
                }
                report['placement_recommendations'].append(recommendation)
        
        # Sort recommendations by confidence
        report['placement_recommendations'].sort(key=lambda x: x['average_confidence'], reverse=True)
        
        return report

def load_cfg_files(cfg_dir: str) -> List[Dict]:
    """Load CFG files from directory"""
    cfg_files = []
    for root, dirs, files in os.walk(cfg_dir):
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

def print_analysis_report(analysis: Dict, predictions: Dict, report: Dict):
    """Print comprehensive analysis report"""
    print("\n" + "="*80)
    print("NODE-LEVEL ANNOTATION TARGET ANALYSIS")
    print("="*80)
    
    print(f"\n📊 CFG Analysis for {analysis.get('method_name', report.get('method_name', 'unknown'))}:")
    print(f"  Total Nodes: {analysis['total_nodes']}")
    print(f"  Annotation Targets: {len(analysis['annotation_targets'])}")
    print(f"  Non-Targets: {analysis['non_targets']}")
    
    print(f"\n🎯 Target Types:")
    for target_type, count in analysis['target_types'].items():
        if count > 0:
            print(f"  {target_type.title()}: {count}")
    
    print(f"\n🤖 Model Predictions:")
    for model_name, targets in predictions.items():
        print(f"  {model_name}: {len(targets)} annotation targets")
        for target in targets:
            print(f"    Line {target.get('line', 'N/A')}: {target.get('annotation_type', 'unknown')} "
                  f"(confidence: {target.get('prediction_score', 0):.3f})")
    
    print(f"\n🎯 Placement Recommendations:")
    for rec in report['placement_recommendations']:
        print(f"  Line {rec['line']}: {rec['annotation_type']} "
              f"({rec['models_count']} models agree, "
              f"confidence: {rec['average_confidence']:.3f}) - {rec['recommendation'].upper()}")
    
    print(f"\n🤝 Model Consensus:")
    if report['model_consensus']:
        for consensus in report['model_consensus']:
            print(f"  Line {consensus['line']}: {', '.join(consensus['models_agreeing'])} agree")
    else:
        print("  No consensus between models")
    
    print("\n" + "="*80)

def main():
    """Main integration function"""
    parser = argparse.ArgumentParser(description='Node-level model integration demo')
    parser.add_argument('--cfg_dir', default='test_results/model_evaluation/cfg_output', 
                       help='Directory containing CFG files')
    parser.add_argument('--models', nargs='+', default=['HGT', 'GBT', 'Causal'],
                       help='Models to use for prediction')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = NodeLevelPipeline()
    
    # Load CFG files
    cfg_files = load_cfg_files(args.cfg_dir)
    if not cfg_files:
        logger.error(f"No CFG files found in {args.cfg_dir}")
        return
    
    logger.info(f"Loaded {len(cfg_files)} CFG files")
    
    # Train models
    training_results = pipeline.train_models(cfg_files)
    logger.info("Training completed")
    
    # Demonstrate on each CFG
    for cfg_file in cfg_files:
        cfg_data = cfg_file['data']
        method_name = cfg_data.get('method_name', 'unknown')
        
        logger.info(f"Processing {method_name}...")
        
        # Analyze annotation targets
        analysis = pipeline.analyze_annotation_targets(cfg_data)
        
        # Get predictions from all models
        predictions = pipeline.predict_annotation_targets(cfg_data, args.models)
        
        # Generate placement report
        report = pipeline.generate_placement_report(predictions, cfg_data)
        
        # Print analysis
        print_analysis_report(analysis, predictions, report)

if __name__ == "__main__":
    main()
