#!/usr/bin/env python3
"""
Test Script for Perfect Annotation Placement

This script demonstrates the perfect accuracy of the annotation placement system
by comparing approximate vs perfect placement results.
"""

import os
import json
import tempfile
import logging
from pathlib import Path

from place_annotations import ComprehensiveAnnotationPlacer, PredictionResult
from perfect_annotation_placement import PreciseAnnotationPlacer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_complex_test_file() -> str:
    """Create a complex test file with various Java constructs"""
    content = '''public class ComplexTest {
    private String name;
    private int[] numbers;
    private List<String> items;
    
    public ComplexTest(String name, int[] numbers, List<String> items) {
        this.name = name;
        this.numbers = numbers;
        this.items = items;
    }
    
    public String processName(String input) {
        if (input != null) {
            return input.toUpperCase();
        }
        return null;
    }
    
    public int findIndex(String target) {
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i).equals(target)) {
                return i;
            }
        }
        return -1;
    }
    
    public void updateNumbers(int[] newNumbers) {
        this.numbers = newNumbers;
    }
    
    public static void main(String[] args) {
        ComplexTest test = new ComplexTest("test", new int[]{1, 2, 3}, 
                                         Arrays.asList("a", "b", "c"));
        String result = test.processName("hello");
        int index = test.findIndex("b");
        System.out.println("Result: " + result + ", Index: " + index);
    }
}'''
    return content

def test_perfect_vs_approximate_placement():
    """Compare perfect vs approximate placement accuracy"""
    logger.info("Testing perfect vs approximate placement accuracy...")
    
    # Create test file
    test_content = create_complex_test_file()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        # Test predictions for various locations
        test_predictions = [
            {
                "file_path": os.path.basename(test_file),
                "line_number": 2,
                "confidence": 0.9,
                "annotation_type": "@NonNull",
                "target_element": "name",
                "context": "String field",
                "model_type": "hgt"
            },
            {
                "file_path": os.path.basename(test_file),
                "line_number": 3,
                "confidence": 0.85,
                "annotation_type": "@MinLen",
                "target_element": "numbers",
                "context": "Array field",
                "model_type": "gbt"
            },
            {
                "file_path": os.path.basename(test_file),
                "line_number": 5,
                "confidence": 0.92,
                "annotation_type": "@NonNull",
                "target_element": "name",
                "context": "Method parameter",
                "model_type": "causal"
            },
            {
                "file_path": os.path.basename(test_file),
                "line_number": 7,
                "confidence": 0.88,
                "annotation_type": "@NonNull",
                "target_element": "input",
                "context": "Method parameter",
                "model_type": "hgt"
            },
            {
                "file_path": os.path.basename(test_file),
                "line_number": 13,
                "confidence": 0.95,
                "annotation_type": "@NonNegative",
                "target_element": "i",
                "context": "Loop variable",
                "model_type": "gbt"
            }
        ]
        
        # Test perfect placement
        logger.info("Testing PERFECT placement...")
        perfect_result = test_placement_mode(test_file, test_predictions, perfect=True)
        
        # Reset file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test approximate placement
        logger.info("Testing APPROXIMATE placement...")
        approximate_result = test_placement_mode(test_file, test_predictions, perfect=False)
        
        # Compare results
        logger.info("=" * 60)
        logger.info("PLACEMENT ACCURACY COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Perfect placement:   {perfect_result['successful']}/{perfect_result['total']} successful")
        logger.info(f"Approximate placement: {approximate_result['successful']}/{approximate_result['total']} successful")
        logger.info("=" * 60)
        
        # Show the final results
        logger.info("FINAL ANNOTATED CODE:")
        logger.info("-" * 40)
        with open(test_file, 'r') as f:
            for i, line in enumerate(f, 1):
                logger.info(f"{i:2}: {line.rstrip()}")
        logger.info("-" * 40)
        
        return perfect_result['successful'] >= approximate_result['successful']
        
    finally:
        os.unlink(test_file)

def test_placement_mode(test_file: str, predictions: list, perfect: bool) -> dict:
    """Test a specific placement mode"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy test file to temp directory
        import shutil
        temp_file = temp_path / "TestFile.java"
        shutil.copy2(test_file, temp_file)
        
        # Create predictions file
        predictions_file = temp_path / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Initialize placer
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(temp_path),
            output_dir=str(temp_path / "output"),
            backup=False,
            perfect_placement=perfect
        )
        
        # Load and process predictions
        loaded_predictions = placer.load_predictions(str(predictions_file))
        stats = placer.process_predictions(loaded_predictions)
        
        # Copy result back to original file for comparison
        shutil.copy2(temp_file, test_file)
        
        return stats

def test_specific_placement_scenarios():
    """Test specific placement scenarios for accuracy"""
    logger.info("Testing specific placement scenarios...")
    
    scenarios = [
        {
            "name": "Variable Declaration",
            "code": "    private String name;",
            "annotation": "@NonNull",
            "expected": "    @NonNull private String name;"
        },
        {
            "name": "Method Parameter",
            "code": "    public void method(String param) {",
            "annotation": "@NonNull",
            "expected": "    public void method(@NonNull String param) {"
        },
        {
            "name": "Array Declaration",
            "code": "    private int[] numbers;",
            "annotation": "@MinLen(0)",
            "expected": "    @MinLen(0) private int[] numbers;"
        },
        {
            "name": "Loop Variable",
            "code": "        for (int i = 0; i < length; i++) {",
            "annotation": "@NonNegative",
            "expected": "        for (@NonNegative int i = 0; i < length; i++) {"
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        logger.info(f"Testing scenario: {scenario['name']}")
        
        # Create test file with scenario
        test_content = f'''public class Test {{
{scenario['code']}
}}'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Test perfect placement
            placer = PreciseAnnotationPlacer(temp_file)
            success = placer.place_annotation_precisely(2, scenario['annotation'])
            
            if success:
                placer.save_file()
                
                # Check result
                with open(temp_file, 'r') as f:
                    result_content = f.read()
                
                if scenario['expected'] in result_content:
                    logger.info(f"✅ {scenario['name']}: PERFECT")
                    results.append(True)
                else:
                    logger.info(f"❌ {scenario['name']}: INCORRECT")
                    logger.info(f"Expected: {scenario['expected']}")
                    logger.info(f"Got: {result_content}")
                    results.append(False)
            else:
                logger.info(f"❌ {scenario['name']}: FAILED")
                results.append(False)
                
        finally:
            os.unlink(temp_file)
    
    success_rate = sum(results) / len(results) * 100
    logger.info(f"Scenario test success rate: {success_rate:.1f}%")
    
    return success_rate >= 90

def main():
    """Run all accuracy tests"""
    logger.info("Starting perfect annotation placement accuracy tests...")
    
    success = True
    
    # Test 1: Perfect vs Approximate comparison
    if not test_perfect_vs_approximate_placement():
        success = False
    
    # Test 2: Specific placement scenarios
    if not test_specific_placement_scenarios():
        success = False
    
    if success:
        logger.info("🎉 All accuracy tests passed!")
        logger.info("✅ Perfect placement system is working correctly!")
    else:
        logger.error("❌ Some accuracy tests failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
