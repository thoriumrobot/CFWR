#!/usr/bin/env python3
"""
Test script for Enhanced Causal Model Integration
Demonstrates usage of the enhanced causal model with existing annotation type scripts
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_causal_positive():
    """Test enhanced causal model with @Positive annotations"""
    logger.info("Testing Enhanced Causal Model with @Positive annotations...")
    
    cmd = [
        'python', 'annotation_type_rl_positive.py',
        '--base_model', 'enhanced_causal',
        '--episodes', '10',
        '--project_root', '/home/ubuntu/checker-framework/checker/tests/index',
        '--learning_rate', '0.001',
        '--hidden_dim', '256',
        '--dropout_rate', '0.3',
        '--device', 'cpu'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Enhanced Causal @Positive test PASSED")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error("❌ Enhanced Causal @Positive test FAILED")
            logger.error(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Enhanced Causal @Positive test TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ Enhanced Causal @Positive test ERROR: {e}")
        return False

def test_enhanced_causal_nonnegative():
    """Test enhanced causal model with @NonNegative annotations"""
    logger.info("Testing Enhanced Causal Model with @NonNegative annotations...")
    
    cmd = [
        'python', 'annotation_type_rl_nonnegative.py',
        '--base_model', 'enhanced_causal',
        '--episodes', '10',
        '--project_root', '/home/ubuntu/checker-framework/checker/tests/index',
        '--learning_rate', '0.001',
        '--hidden_dim', '256',
        '--dropout_rate', '0.3',
        '--device', 'cpu'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Enhanced Causal @NonNegative test PASSED")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error("❌ Enhanced Causal @NonNegative test FAILED")
            logger.error(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Enhanced Causal @NonNegative test TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ Enhanced Causal @NonNegative test ERROR: {e}")
        return False

def test_enhanced_causal_gtenegativeone():
    """Test enhanced causal model with @GTENegativeOne annotations"""
    logger.info("Testing Enhanced Causal Model with @GTENegativeOne annotations...")
    
    cmd = [
        'python', 'annotation_type_rl_gtenegativeone.py',
        '--base_model', 'enhanced_causal',
        '--episodes', '10',
        '--project_root', '/home/ubuntu/checker-framework/checker/tests/index',
        '--learning_rate', '0.001',
        '--hidden_dim', '256',
        '--dropout_rate', '0.3',
        '--device', 'cpu'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Enhanced Causal @GTENegativeOne test PASSED")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error("❌ Enhanced Causal @GTENegativeOne test FAILED")
            logger.error(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Enhanced Causal @GTENegativeOne test TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ Enhanced Causal @GTENegativeOne test ERROR: {e}")
        return False

def test_original_causal_comparison():
    """Test original causal model for comparison"""
    logger.info("Testing Original Causal Model for comparison...")
    
    cmd = [
        'python', 'annotation_type_rl_positive.py',
        '--base_model', 'causal',
        '--episodes', '10',
        '--project_root', '/home/ubuntu/checker-framework/checker/tests/index',
        '--learning_rate', '0.001',
        '--device', 'cpu'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Original Causal test PASSED")
            return True
        else:
            logger.error("❌ Original Causal test FAILED")
            logger.error(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Original Causal test TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ Original Causal test ERROR: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Enhanced Causal Model Integration Tests")
    logger.info("=" * 60)
    
    # Check if enhanced causal model is available
    try:
        from enhanced_causal_model import EnhancedCausalModel
        logger.info("✅ Enhanced Causal Model is available")
    except ImportError as e:
        logger.error(f"❌ Enhanced Causal Model not available: {e}")
        return False
    
    tests = [
        ("Enhanced Causal @Positive", test_enhanced_causal_positive),
        ("Enhanced Causal @NonNegative", test_enhanced_causal_nonnegative),
        ("Enhanced Causal @GTENegativeOne", test_enhanced_causal_gtenegativeone),
        ("Original Causal Comparison", test_original_causal_comparison)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name}...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced Causal Model integration successful!")
        return True
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed. Please check the logs above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
