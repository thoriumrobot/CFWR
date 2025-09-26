#!/usr/bin/env python3
"""
Detailed Analysis of Model Prediction Counts and Class Diversity Issues

This script analyzes why different models produce different prediction counts
and what causes the GBT class diversity issues.
"""

import os
import json
import numpy as np
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelAnalysis:
    """Analyzes model prediction counts and class diversity issues"""
    
    def __init__(self, cfg_dir: str):
        self.cfg_dir = cfg_dir
        
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
    
    def analyze_hgt_data_processing(self, cfg_files: List[Dict]) -> Dict:
        """Analyze how HGT processes data"""
        logger.info("Analyzing HGT data processing...")
        
        try:
            from hgt import create_heterodata
            
            total_nodes = 0
            successful_cfgs = 0
            failed_cfgs = 0
            
            for cfg_file in cfg_files:
                try:
                    # Create heterodata
                    heterodata = create_heterodata(cfg_file['data'])
                    if heterodata is None:
                        failed_cfgs += 1
                        continue
                    
                    node_count = heterodata['node'].x.shape[0]
                    total_nodes += node_count
                    successful_cfgs += 1
                    
                    logger.info(f"HGT - {cfg_file['method']}: {node_count} nodes")
                    
                except Exception as e:
                    logger.warning(f"HGT error processing {cfg_file['method']}: {e}")
                    failed_cfgs += 1
            
            return {
                'model': 'HGT',
                'total_nodes': total_nodes,
                'successful_cfgs': successful_cfgs,
                'failed_cfgs': failed_cfgs,
                'avg_nodes_per_cfg': total_nodes / successful_cfgs if successful_cfgs > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"HGT analysis failed: {e}")
            return {'model': 'HGT', 'error': str(e)}
    
    def analyze_gbt_data_processing(self, cfg_files: List[Dict]) -> Dict:
        """Analyze how GBT processes data and class diversity"""
        logger.info("Analyzing GBT data processing...")
        
        try:
            from gbt import extract_features_from_cfg
            
            total_features = 0
            successful_extractions = 0
            failed_extractions = 0
            all_labels = []
            
            for cfg_file in cfg_files:
                try:
                    # Extract features
                    features = extract_features_from_cfg(cfg_file['data'])
                    if len(features) != 8:  # GBT expects 8 features
                        # Pad or truncate features to match expected size
                        if len(features) < 8:
                            features.extend([0.0] * (8 - len(features)))
                        else:
                            features = features[:8]
                    
                    # Create synthetic labels based on features
                    complexity = sum(features[2:])  # Sum of control flow features
                    # Create more diverse labels to avoid single class issue
                    if complexity > 3:
                        label = 1
                    elif complexity > 1:
                        label = 1
                    else:
                        label = 0
                    
                    all_labels.append(label)
                    total_features += 1
                    successful_extractions += 1
                    
                    logger.info(f"GBT - {cfg_file['method']}: features={features}, complexity={complexity}, label={label}")
                    
                except Exception as e:
                    logger.warning(f"GBT error processing {cfg_file['method']}: {e}")
                    failed_extractions += 1
                    # Create dummy features if extraction fails
                    all_labels.append(0)  # Default label
                    total_features += 1
            
            # Analyze class diversity
            unique_labels = set(all_labels)
            label_counts = {label: all_labels.count(label) for label in unique_labels}
            
            return {
                'model': 'GBT',
                'total_features': total_features,
                'successful_extractions': successful_extractions,
                'failed_extractions': failed_extractions,
                'unique_labels': list(unique_labels),
                'label_counts': label_counts,
                'class_diversity_issue': len(unique_labels) < 2,
                'all_labels': all_labels
            }
            
        except Exception as e:
            logger.error(f"GBT analysis failed: {e}")
            return {'model': 'GBT', 'error': str(e)}
    
    def analyze_causal_data_processing(self, cfg_files: List[Dict]) -> Dict:
        """Analyze how Causal model processes data"""
        logger.info("Analyzing Causal data processing...")
        
        try:
            from causal_model import extract_features_and_labels
            
            total_features = 0
            successful_extractions = 0
            failed_extractions = 0
            all_labels = []
            
            for cfg_file in cfg_files:
                try:
                    # Extract features
                    features_list = extract_features_and_labels(cfg_file['data'], [])
                    if not features_list:
                        # Create dummy features if extraction fails
                        all_labels.append(0)  # Default label
                        total_features += 1
                        continue
                    
                    for features in features_list:
                        if len(features) == 12:  # Causal model expects 12 features
                            # Create synthetic label based on features
                            complexity = sum(features[2:])  # Sum of control flow features
                            label = 1 if complexity > 2 else 0
                            
                            all_labels.append(label)
                            total_features += 1
                            successful_extractions += 1
                            
                            logger.info(f"Causal - {cfg_file['method']}: features={features}, complexity={complexity}, label={label}")
                        else:
                            # Pad or truncate features to match expected size
                            if len(features) < 12:
                                features = list(features) + [0.0] * (12 - len(features))
                            else:
                                features = list(features)[:12]
                            
                            complexity = sum(features[2:])
                            label = 1 if complexity > 2 else 0
                            
                            all_labels.append(label)
                            total_features += 1
                            successful_extractions += 1
                    
                except Exception as e:
                    logger.warning(f"Causal error processing {cfg_file['method']}: {e}")
                    failed_extractions += 1
                    # Create dummy features if extraction fails
                    all_labels.append(0)  # Default label
                    total_features += 1
            
            # Analyze class diversity
            unique_labels = set(all_labels)
            label_counts = {label: all_labels.count(label) for label in unique_labels}
            
            return {
                'model': 'Causal',
                'total_features': total_features,
                'successful_extractions': successful_extractions,
                'failed_extractions': failed_extractions,
                'unique_labels': list(unique_labels),
                'label_counts': label_counts,
                'class_diversity_issue': len(unique_labels) < 2,
                'all_labels': all_labels
            }
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}")
            return {'model': 'Causal', 'error': str(e)}
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive analysis of all models"""
        logger.info("Starting comprehensive model analysis...")
        
        # Load test data
        cfg_files = self.load_cfg_data()
        logger.info(f"Loaded {len(cfg_files)} CFG files for analysis")
        
        if not cfg_files:
            logger.error("No CFG files found for analysis")
            return {}
        
        # Analyze each model
        results = {}
        results['HGT'] = self.analyze_hgt_data_processing(cfg_files)
        results['GBT'] = self.analyze_gbt_data_processing(cfg_files)
        results['Causal'] = self.analyze_causal_data_processing(cfg_files)
        
        return results
    
    def print_analysis_summary(self, results: Dict):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("MODEL PREDICTION COUNT AND CLASS DIVERSITY ANALYSIS")
        print("="*80)
        
        for model_name in ['HGT', 'GBT', 'Causal']:
            if model_name in results:
                result = results[model_name]
                if 'error' in result:
                    print(f"\n{model_name} Model:")
                    print(f"  Status: FAILED")
                    print(f"  Error: {result['error']}")
                else:
                    print(f"\n{model_name} Model:")
                    if model_name == 'HGT':
                        print(f"  Total Nodes Processed: {result.get('total_nodes', 0)}")
                        print(f"  Successful CFGs: {result.get('successful_cfgs', 0)}")
                        print(f"  Failed CFGs: {result.get('failed_cfgs', 0)}")
                        print(f"  Average Nodes per CFG: {result.get('avg_nodes_per_cfg', 0):.2f}")
                    else:
                        print(f"  Total Features Processed: {result.get('total_features', 0)}")
                        print(f"  Successful Extractions: {result.get('successful_extractions', 0)}")
                        print(f"  Failed Extractions: {result.get('failed_extractions', 0)}")
                        print(f"  Unique Labels: {result.get('unique_labels', [])}")
                        print(f"  Label Counts: {result.get('label_counts', {})}")
                        print(f"  Class Diversity Issue: {result.get('class_diversity_issue', False)}")
                        print(f"  All Labels: {result.get('all_labels', [])}")
        
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"  1. HGT processes at the NODE level (one prediction per CFG node)")
        print(f"  2. GBT processes at the CFG level (one prediction per CFG)")
        print(f"  3. Causal processes at the FEATURE level (one prediction per feature set)")
        print(f"  4. Different processing levels explain different prediction counts")
        print(f"  5. GBT class diversity issues occur when all labels are the same")
        
        print("\n" + "="*80)

def main():
    """Main analysis function"""
    analyzer = ModelAnalysis(cfg_dir="test_results/model_evaluation/cfg_output")
    results = analyzer.run_comprehensive_analysis()
    analyzer.print_analysis_summary(results)

if __name__ == "__main__":
    main()
