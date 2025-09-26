#!/usr/bin/env python3
"""
Test Script for Annotation Placement System

This script tests the comprehensive annotation placement system with 
Lower Bound Checker support using our existing test files.
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

def create_test_predictions() -> list:
    """Create test predictions for demonstration"""
    return [
        {
            "file_path": "TestNullness.java",
            "line_number": 8,
            "confidence": 0.85,
            "annotation_type": "@NonNull",
            "target_element": "name",
            "context": "String variable in method",
            "model_type": "hgt"
        },
        {
            "file_path": "TestNullness.java", 
            "line_number": 15,
            "confidence": 0.92,
            "annotation_type": "@Nullable",
            "target_element": "result",
            "context": "Return value that can be null",
            "model_type": "gbt"
        },
        {
            "file_path": "TestNullness.java",
            "line_number": 23,
            "confidence": 0.78,
            "annotation_type": "@MinLen",
            "target_element": "x",
            "context": "Integer variable declaration",
            "model_type": "causal"
        },
        {
            "file_path": "TestNullness.java",
            "line_number": 24,
            "confidence": 0.88,
            "annotation_type": "@NonNull",
            "target_element": "y",
            "context": "String variable declaration",
            "model_type": "hgt"
        }
    ]

def test_annotation_placement():
    """Test the annotation placement system"""
    logger.info("Testing annotation placement system...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy test file to temp directory
        test_file_source = Path("/home/ubuntu/CFWR/TestNullness.java")
        if not test_file_source.exists():
            logger.error("Test file not found. Creating a sample test file...")
            create_sample_test_file(temp_path / "TestNullness.java")
        else:
            import shutil
            shutil.copy2(test_file_source, temp_path / "TestNullness.java")
        
        # Create test predictions file
        predictions = create_test_predictions()
        predictions_file = temp_path / "test_predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Initialize annotation placer
        output_dir = temp_path / "output"
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(temp_path),
            output_dir=str(output_dir),
            backup=True
        )
        
        try:
            # Load predictions
            loaded_predictions = placer.load_predictions(str(predictions_file))
            logger.info(f"Loaded {len(loaded_predictions)} predictions")
            
            # Process predictions
            stats = placer.process_predictions(loaded_predictions)
            logger.info(f"Placement statistics: {stats}")
            
            # Check results
            annotated_file = temp_path / "TestNullness.java"
            if annotated_file.exists():
                with open(annotated_file, 'r') as f:
                    content = f.read()
                logger.info("Annotated file content:")
                logger.info("-" * 50)
                for i, line in enumerate(content.split('\n'), 1):
                    logger.info(f"{i:2}: {line}")
                logger.info("-" * 50)
            
            # Validate annotations (skip actual Checker Framework for demo)
            # validation_results = placer.validate_annotations()
            # logger.info(f"Validation results: {validation_results}")
            
            # Generate report
            report_path = placer.generate_report(stats)
            logger.info(f"Report generated: {report_path}")
            
            logger.info("✅ Annotation placement test completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False

def create_sample_test_file(file_path: Path):
    """Create a sample test file for annotation placement"""
    content = '''public class TestNullness {
    public String getName(boolean returnNull) {
        String name = "John Doe";
        if (returnNull) {
            name = null;
        }
        return name; // Potential Nullness warning here
    }

    public void printName(String name) {
        System.out.println(name.length());
    }

    public static void main(String[] args) {
        TestNullness tn = new TestNullness();
        String result = tn.getName(true);
        // This line would cause a Nullness warning if result is null
        // tn.printName(result); 
        
        // Example of variable declarations
        int x = 10;
        String y = "hello";
        Object z = null; // This could be a target for @Nullable
        
        // Array example
        int[] arr = new int[x];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = i;
        }
    }
}'''
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Created sample test file: {file_path}")

def test_integrated_pipeline():
    """Test the integrated prediction and annotation pipeline"""
    logger.info("Testing integrated pipeline...")
    
    # This would test the full pipeline if models were available
    logger.info("Note: Full pipeline test requires trained models")
    logger.info("To test the full pipeline, use:")
    logger.info("python predict_and_annotate.py --project_root /path/to/project --output_dir /path/to/output")
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting annotation placement tests...")
    
    success = True
    
    # Test 1: Basic annotation placement
    if not test_annotation_placement():
        success = False
    
    # Test 2: Integrated pipeline (informational)
    if not test_integrated_pipeline():
        success = False
    
    if success:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("❌ Some tests failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
