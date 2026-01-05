#!/usr/bin/env python3
"""
Debug script to test enhanced causal feature extraction
"""

import json
import os
from enhanced_causal_model import extract_enhanced_causal_features

def test_feature_extraction():
    """Test feature extraction with sample data"""
    
    # Sample node data
    sample_node = {
        'id': 1,
        'label': 'count = size + 1',
        'node_type': 'variable',
        'line': 10
    }
    
    # Sample CFG data
    sample_cfg = {
        'nodes': [
            {'id': 1, 'label': 'count = size + 1', 'node_type': 'variable', 'line': 10},
            {'id': 2, 'label': 'if (count > 0)', 'node_type': 'method', 'line': 11}
        ],
        'control_edges': [
            {'source': 1, 'target': 2}
        ],
        'dataflow_edges': [
            {'source': 1, 'target': 2}
        ]
    }
    
    print("Testing enhanced causal feature extraction...")
    
    try:
        features = extract_enhanced_causal_features(sample_node, sample_cfg)
        print(f"✅ Feature extraction successful!")
        print(f"📊 Number of features: {len(features)}")
        print(f"📋 Features: {features}")
        
        if len(features) == 32:
            print("✅ Correct number of features (32)")
        else:
            print(f"❌ Wrong number of features: {len(features)} (expected 32)")
            
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_extraction()
