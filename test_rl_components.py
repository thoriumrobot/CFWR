#!/usr/bin/env python3
"""
Simple test script for RL training components
"""

import os
import tempfile
import shutil
from enhanced_rl_training import EnhancedReinforcementLearningTrainer
from annotation_placement import AnnotationPlacementManager
from checker_framework_integration import CheckerFrameworkEvaluator, CheckerType

def test_rl_components():
    """Test the RL training components"""
    print("Testing RL Training Components...")
    
    # Test 1: Annotation Placement Manager
    print("\n1. Testing Annotation Placement Manager...")
    try:
        manager = AnnotationPlacementManager("test_dataflow.java")
        structure = manager.get_code_structure()
        print(f"   ✓ Found {len(structure['variables'])} variables, {len(structure['methods'])} methods")
        
        # Test annotation placement
        success = manager.place_annotations([3, 4], 'nullness')
        print(f"   ✓ Annotation placement: {'Success' if success else 'Failed'}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Checker Framework Evaluator
    print("\n2. Testing Checker Framework Evaluator...")
    try:
        evaluator = CheckerFrameworkEvaluator()
        result = evaluator.evaluate_file("test_dataflow.java", CheckerType.NULLNESS)
        print(f"   ✓ Evaluation: {'Success' if result.success else 'Failed'}")
        print(f"   ✓ Found {len(result.original_warnings)} warnings")
        print(f"   ✓ Reward: {evaluator.get_reward(result):.3f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: RL Trainer Initialization
    print("\n3. Testing RL Trainer Initialization...")
    try:
        trainer = EnhancedReinforcementLearningTrainer(
            model_type='hgt',
            learning_rate=0.001,
            device='cpu',
            checker_type='nullness',
            reward_strategy='adaptive'
        )
        print(f"   ✓ Trainer initialized with {trainer.model_type} model")
        print(f"   ✓ Checker type: {trainer.checker_type.value}")
        print(f"   ✓ Reward strategy: {trainer.reward_strategy}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Model Prediction
    print("\n4. Testing Model Prediction...")
    try:
        # Create a simple CFG data structure
        cfg_data = {
            'method_name': 'testMethod',
            'java_file': 'test_dataflow.java',
            'nodes': [
                {'id': 0, 'label': 'Entry', 'line': 1, 'node_type': 'control'},
                {'id': 1, 'label': 'LocalVariableDeclaration', 'line': 3, 'node_type': 'control'},
                {'id': 2, 'label': 'Exit', 'line': 9, 'node_type': 'control'}
            ],
            'control_edges': [
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2}
            ],
            'dataflow_edges': [
                {'source': 1, 'target': 1, 'variable': 'x'}
            ]
        }
        
        predicted_lines = trainer.predict_annotation_locations(cfg_data)
        print(f"   ✓ Predicted {len(predicted_lines)} annotation locations: {predicted_lines}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✓ All component tests completed!")

if __name__ == '__main__':
    test_rl_components()
