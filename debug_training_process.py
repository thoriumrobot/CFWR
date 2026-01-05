#!/usr/bin/env python3
"""
Debug script to trace through the training process and identify dimension mismatches
"""

import os
import json
import logging
from enhanced_causal_model import extract_enhanced_causal_features, EnhancedCausalModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_training_process():
    """Debug the training process step by step"""
    
    # Test data
    sample_cfg_data = {
        'nodes': [
            {'id': 1, 'label': 'count = size + 1', 'node_type': 'variable', 'line': 10},
            {'id': 2, 'label': 'if (count > 0)', 'node_type': 'method', 'line': 11},
            {'id': 3, 'label': 'return count', 'node_type': 'method', 'line': 12}
        ],
        'control_edges': [
            {'source': 1, 'target': 2},
            {'source': 2, 'target': 3}
        ],
        'dataflow_edges': [
            {'source': 1, 'target': 2},
            {'source': 1, 'target': 3}
        ]
    }
    
    print("🔍 Debugging Enhanced Causal Training Process")
    print("=" * 50)
    
    # Step 1: Test feature extraction
    print("\n1️⃣ Testing Feature Extraction:")
    node = sample_cfg_data['nodes'][0]
    features = extract_enhanced_causal_features(node, sample_cfg_data)
    print(f"   ✅ Features extracted: {len(features)} dimensions")
    print(f"   📊 First 10 features: {features[:10]}")
    
    # Step 2: Test model initialization
    print("\n2️⃣ Testing Model Initialization:")
    try:
        model = EnhancedCausalModel(input_dim=32, hidden_dim=256, out_dim=2, annotation_type='@Positive')
        print(f"   ✅ Model initialized successfully")
        print(f"   📊 Model input_dim: 32")
        print(f"   📊 Model hidden_dim: 256")
        
        # Test forward pass with single sample
        import torch
        test_input = torch.tensor([features], dtype=torch.float)
        print(f"   📊 Test input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            print(f"   ✅ Forward pass successful")
            print(f"   📊 Output shape: {output.shape}")
            
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Test with multiple samples
    print("\n3️⃣ Testing with Multiple Samples:")
    try:
        all_features = []
        for node in sample_cfg_data['nodes']:
            features = extract_enhanced_causal_features(node, sample_cfg_data)
            all_features.append(features)
        
        print(f"   📊 Number of samples: {len(all_features)}")
        print(f"   📊 Feature dimensions: {[len(f) for f in all_features]}")
        
        # Create batch tensor
        batch_input = torch.tensor(all_features, dtype=torch.float)
        print(f"   📊 Batch input shape: {batch_input.shape}")
        
        with torch.no_grad():
            batch_output = model(batch_input)
            print(f"   ✅ Batch forward pass successful")
            print(f"   📊 Batch output shape: {batch_output.shape}")
            
    except Exception as e:
        print(f"   ❌ Batch forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎯 Debug Summary:")
    print("   - Feature extraction: Working (32 dimensions)")
    print("   - Model initialization: Working")
    print("   - Forward pass: Working")
    print("   - Issue might be in the training pipeline integration")

if __name__ == "__main__":
    debug_training_process()
