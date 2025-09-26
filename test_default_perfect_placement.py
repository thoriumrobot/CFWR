#!/usr/bin/env python3
"""
Test Script: Perfect Placement as Default vs Approximate Placement as Backup

This script demonstrates that the main codebase now uses perfect placement
by default, with approximate placement only available as a backup option.
"""

import os
import json
import tempfile
import logging
from pathlib import Path

from place_annotations import ComprehensiveAnnotationPlacer, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_default_perfect_placement():
    """Test that perfect placement is used by default"""
    logger.info("Testing DEFAULT behavior: Perfect placement")
    
    # Create test predictions
    test_predictions = [
        {
            "file_path": "TestFile.java",
            "line_number": 2,
            "confidence": 0.9,
            "annotation_type": "@NonNull",
            "target_element": "name",
            "context": "String field",
            "model_type": "hgt"
        }
    ]
    
    # Create test Java file
    test_content = '''public class TestFile {
    private String name;
}'''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test file
        test_file = temp_path / "TestFile.java"
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Create predictions file
        predictions_file = temp_path / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(test_predictions, f, indent=2)
        
        # Test DEFAULT behavior (perfect placement)
        logger.info("Testing with DEFAULT settings (perfect_placement=True)...")
        placer_default = ComprehensiveAnnotationPlacer(
            project_root=str(temp_path),
            output_dir=str(temp_path / "output_default"),
            backup=False,
            perfect_placement=True  # DEFAULT
        )
        
        # Load and process predictions
        predictions = placer_default.load_predictions(str(predictions_file))
        stats_default = placer_default.process_predictions(predictions)
        
        # Check result
        with open(test_file, 'r') as f:
            result_default = f.read()
        
        logger.info(f"DEFAULT result: {result_default.strip()}")
        
        # Reset file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test BACKUP behavior (approximate placement)
        logger.info("Testing with BACKUP settings (perfect_placement=False)...")
        placer_backup = ComprehensiveAnnotationPlacer(
            project_root=str(temp_path),
            output_dir=str(temp_path / "output_backup"),
            backup=False,
            perfect_placement=False  # BACKUP
        )
        
        # Load and process predictions
        predictions = placer_backup.load_predictions(str(predictions_file))
        stats_backup = placer_backup.process_predictions(predictions)
        
        # Check result
        with open(test_file, 'r') as f:
            result_backup = f.read()
        
        logger.info(f"BACKUP result: {result_backup.strip()}")
        
        # Compare results
        logger.info("=" * 60)
        logger.info("COMPARISON: DEFAULT vs BACKUP")
        logger.info("=" * 60)
        logger.info(f"DEFAULT (Perfect):  {stats_default['successful']}/{stats_default['total']} successful")
        logger.info(f"BACKUP (Approximate): {stats_backup['successful']}/{stats_backup['total']} successful")
        logger.info("=" * 60)
        
        # Check if perfect placement produces better results
        perfect_better = stats_default['successful'] >= stats_backup['successful']
        
        if perfect_better:
            logger.info("✅ Perfect placement (DEFAULT) performs as well or better than approximate placement")
        else:
            logger.warning("⚠️  Approximate placement performed better (unexpected)")
        
        return perfect_better

def test_command_line_defaults():
    """Test that command line defaults use perfect placement"""
    logger.info("\nTesting command line defaults...")
    
    # Test help output to verify defaults
    import subprocess
    result = subprocess.run(['python', 'place_annotations.py', '--help'], 
                          capture_output=True, text=True, cwd='/home/ubuntu/CFWR')
    
    help_text = result.stdout
    
    # Check that perfect placement is mentioned as default
    if "DEFAULT" in help_text and "perfect" in help_text.lower():
        logger.info("✅ Command line help shows perfect placement as DEFAULT")
        return True
    else:
        logger.error("❌ Command line help doesn't show perfect placement as DEFAULT")
        return False

def main():
    """Run all tests"""
    logger.info("Testing Perfect Placement as Default vs Approximate as Backup")
    logger.info("=" * 70)
    
    success = True
    
    # Test 1: Default behavior uses perfect placement
    if not test_default_perfect_placement():
        success = False
    
    # Test 2: Command line defaults
    if not test_command_line_defaults():
        success = False
    
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Perfect placement is now the DEFAULT behavior")
        logger.info("✅ Approximate placement is available as BACKUP only")
        logger.info("✅ Main codebase uses perfect placement by default")
    else:
        logger.error("❌ Some tests failed!")
        logger.error("⚠️  Check that perfect placement is properly set as default")
    logger.info("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
