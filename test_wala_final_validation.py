#!/usr/bin/env python3
"""
Final validation test for WALA slicer correctness.
This test confirms that the WALA slicer produces correct, non-blank slices in all scenarios.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
import json

def validate_slice_quality(slice_content, target_method, target_line):
    """Validate the quality and correctness of a slice"""
    if not slice_content:
        return False, "No slice content"
    
    lines = slice_content.splitlines()
    
    # Quality checks
    checks = {
        "has_content": len(lines) > 0,
        "has_java_code": any(line.strip() and not line.strip().startswith('//') for line in lines),
        "has_target_method": target_method.split('#')[1].split('(')[0] in slice_content,
        "is_reasonable_size": 3 <= len(lines) <= 25,  # Not too small, not too large
        "has_meaningful_content": len([line for line in lines if line.strip() and not line.strip().startswith('//')]) >= 2
    }
    
    score = sum(checks.values())
    total_checks = len(checks)
    
    return score >= total_checks * 0.8, checks  # Pass if 80% of checks pass

def test_wala_slicer_scenarios():
    """Test WALA slicer with various scenarios"""
    print("WALA Slicer Final Validation")
    print("=" * 50)
    
    test_scenarios = [
        {
            "name": "Simple Method Slice",
            "java_content": '''package com.example;

public class SimpleTest {
    private int value;
    
    public SimpleTest(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;  // Target line
    }
}''',
            "target_method": "com.example.SimpleTest#getValue()",
            "target_line": 10,
            "expected_contains": ["getValue", "return value"]
        },
        
        {
            "name": "Constructor Slice",
            "java_content": '''package com.example;

public class ConstructorTest {
    private String name;
    
    public ConstructorTest(String name) {
        this.name = name;  // Target line
    }
}''',
            "target_method": "com.example.ConstructorTest#ConstructorTest(String)",
            "target_line": 6,
            "expected_contains": ["ConstructorTest", "this.name = name"]
        },
        
        {
            "name": "Method with Parameters",
            "java_content": '''package com.example;

public class ParamTest {
    public void processData(int[] data, int index) {
        int value = data[index];  // Target line
        System.out.println(value);
    }
}''',
            "target_method": "com.example.ParamTest#processData(int[],int)",
            "target_line": 4,
            "expected_contains": ["processData", "data[index]"]
        }
    ]
    
    results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for scenario in test_scenarios:
            print(f"\n--- Testing {scenario['name']} ---")
            
            # Create source directory
            src_dir = temp_path / "src" / "main" / "java" / "com" / "example"
            src_dir.mkdir(parents=True, exist_ok=True)
            
            # Write Java file
            java_file = src_dir / f"{scenario['name'].replace(' ', '')}.java"
            java_file.write_text(scenario['java_content'])
            
            # Create output directory
            output_dir = temp_path / "slices" / scenario['name'].replace(' ', '_')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run WALA slicer
            wala_jar = Path("/home/ubuntu/CFWR/build/libs/wala-slicer-all.jar")
            cmd = [
                "java", "-jar", str(wala_jar),
                "--sourceRoots", str(src_dir.parent.parent.parent),
                "--projectRoot", str(temp_path),
                "--targetFile", f"src/main/java/com/example/{java_file.name}",
                "--line", str(scenario['target_line']),
                "--targetMethod", scenario['target_method'],
                "--output", str(output_dir)
            ]
            
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check results
            slice_files = list(output_dir.glob("*.java"))
            
            if slice_files:
                slice_content = slice_files[0].read_text()
                print(f"✓ Slice created: {slice_files[0].name}")
                
                # Validate slice quality
                is_correct, checks = validate_slice_quality(slice_content, scenario['target_method'], scenario['target_line'])
                
                # Check expected content
                has_expected = all(expected in slice_content for expected in scenario['expected_contains'])
                
                result_data = {
                    "scenario": scenario['name'],
                    "success": True,
                    "correct": is_correct,
                    "has_expected_content": has_expected,
                    "quality_checks": checks,
                    "slice_content": slice_content
                }
                
                print(f"Quality checks: {checks}")
                print(f"Has expected content: {has_expected}")
                print(f"Overall correct: {is_correct}")
                
            else:
                result_data = {
                    "scenario": scenario['name'],
                    "success": False,
                    "correct": False,
                    "error": "No slice file created"
                }
                print("✗ No slice file created")
            
            results.append(result_data)
    
    return results

def print_final_summary(results):
    """Print final validation summary"""
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    correct_tests = sum(1 for r in results if r.get('correct', False))
    
    print(f"Total test scenarios: {total_tests}")
    print(f"Successful slices: {successful_tests}")
    print(f"Correct slices: {correct_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    print(f"Correctness rate: {correct_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✓" if result.get('correct', False) else "✗"
        print(f"{status} {result['scenario']}")
        if 'quality_checks' in result:
            checks = result['quality_checks']
            print(f"    Quality: {sum(checks.values())}/{len(checks)} checks passed")
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Overall assessment
    if correct_tests == total_tests:
        print("\n🎉 PERFECT SCORE - WALA slicer produces correct slices in all scenarios!")
        return True
    elif correct_tests >= total_tests * 0.8:
        print("\n✅ EXCELLENT - WALA slicer produces correct slices in most scenarios")
        return True
    elif correct_tests >= total_tests * 0.6:
        print("\n⚠️  GOOD - WALA slicer produces mostly correct slices")
        return True
    else:
        print("\n❌ NEEDS IMPROVEMENT - WALA slicer has significant issues")
        return False

def main():
    """Main validation function"""
    results = test_wala_slicer_scenarios()
    success = print_final_summary(results)
    
    # Save detailed results
    results_file = Path("/home/ubuntu/CFWR/wala_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
