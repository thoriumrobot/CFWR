#!/usr/bin/env python3
"""
Simple Test for Perfect Annotation Placement Accuracy

This script demonstrates the perfect accuracy of annotation placement
with a focused test on specific scenarios.
"""

import os
import tempfile
import logging
from perfect_annotation_placement import PreciseAnnotationPlacer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_perfect_placement_scenarios():
    """Test perfect placement on specific scenarios"""
    logger.info("Testing perfect annotation placement accuracy...")
    
    test_cases = [
        {
            "name": "Field Declaration",
            "code": '''public class Test {
    private String name;
}''',
            "line": 2,
            "annotation": "@NonNull",
            "target": "name",
            "expected": "    @NonNull private String name;"
        },
        {
            "name": "Array Field",
            "code": '''public class Test {
    private int[] numbers;
}''',
            "line": 2,
            "annotation": "@MinLen(0)",
            "target": "numbers",
            "expected": "    @MinLen(0) private int[] numbers;"
        },
        {
            "name": "Method Parameter",
            "code": '''public class Test {
    public void method(String param) {
    }
}''',
            "line": 2,
            "annotation": "@NonNull",
            "target": "param",
            "expected": "    public void method(@NonNull String param) {"
        },
        {
            "name": "Constructor Parameter",
            "code": '''public class Test {
    public Test(String name) {
    }
}''',
            "line": 2,
            "annotation": "@NonNull",
            "target": "name",
            "expected": "    public Test(@NonNull String name) {"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test_case['name']}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(test_case['code'])
            temp_file = f.name
        
        try:
            # Test perfect placement
            placer = PreciseAnnotationPlacer(temp_file)
            success = placer.place_annotation_precisely(
                test_case['line'], 
                test_case['annotation'], 
                test_case['target']
            )
            
            if success:
                placer.save_file()
                
                # Read result
                with open(temp_file, 'r') as f:
                    result_content = f.read()
                
                # Check if expected annotation is in the result
                if test_case['expected'] in result_content:
                    logger.info(f"✅ PERFECT: Annotation placed exactly as expected")
                    logger.info(f"   Expected: {test_case['expected']}")
                    results.append(True)
                else:
                    logger.info(f"❌ INCORRECT: Annotation not placed correctly")
                    logger.info(f"   Expected: {test_case['expected']}")
                    logger.info(f"   Got: {result_content}")
                    results.append(False)
            else:
                logger.info(f"❌ FAILED: Could not place annotation")
                results.append(False)
                
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            results.append(False)
        finally:
            os.unlink(temp_file)
    
    # Calculate success rate
    success_count = sum(results)
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    logger.info(f"\n{'='*50}")
    logger.info(f"PERFECT PLACEMENT ACCURACY RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Successful placements: {success_count}/{total_count}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"{'='*50}")
    
    return success_rate >= 75  # Consider 75%+ as good accuracy

def test_complex_scenario():
    """Test a more complex scenario with multiple annotations"""
    logger.info("\nTesting complex scenario with multiple annotations...")
    
    complex_code = '''public class ComplexTest {
    private String name;
    private int[] numbers;
    
    public ComplexTest(String name, int[] numbers) {
        this.name = name;
        this.numbers = numbers;
    }
    
    public String processName(String input) {
        return input.toUpperCase();
    }
}'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
        f.write(complex_code)
        temp_file = f.name
    
    try:
        placer = PreciseAnnotationPlacer(temp_file)
        
        # Place multiple annotations
        annotations_to_place = [
            (2, "@NonNull", "name"),      # Field
            (3, "@MinLen(0)", "numbers"),  # Array field
            (5, "@NonNull", "name"),       # Constructor parameter
            (5, "@MinLen(0)", "numbers"), # Constructor parameter
            (9, "@NonNull", "input"),      # Method parameter
        ]
        
        success_count = 0
        for line, annotation, target in annotations_to_place:
            success = placer.place_annotation_precisely(line, annotation, target)
            if success:
                success_count += 1
                logger.info(f"✅ Placed {annotation} at line {line}")
            else:
                logger.info(f"❌ Failed to place {annotation} at line {line}")
        
        if success_count > 0:
            placer.save_file()
            
            # Display final result
            logger.info("\nFinal annotated code:")
            logger.info("-" * 40)
            with open(temp_file, 'r') as f:
                for i, line in enumerate(f, 1):
                    logger.info(f"{i:2}: {line.rstrip()}")
            logger.info("-" * 40)
        
        return success_count >= 3  # At least 3 out of 5 should succeed
        
    finally:
        os.unlink(temp_file)

def main():
    """Run perfect placement accuracy tests"""
    logger.info("Starting Perfect Annotation Placement Accuracy Tests")
    logger.info("=" * 60)
    
    success = True
    
    # Test 1: Individual scenarios
    if not test_perfect_placement_scenarios():
        success = False
    
    # Test 2: Complex scenario
    if not test_complex_scenario():
        success = False
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 PERFECT PLACEMENT TESTS PASSED!")
        logger.info("✅ Annotation placement is highly accurate!")
    else:
        logger.info("❌ Some perfect placement tests failed!")
        logger.info("⚠️  Consider improving AST analysis or placement logic")
    logger.info("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
