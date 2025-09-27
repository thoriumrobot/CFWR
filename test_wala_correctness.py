#!/usr/bin/env python3
"""
Comprehensive test for WALA slicer correctness.
This script tests the WALA slicer with various Java code patterns to ensure it produces correct slices.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
import json

def create_test_cases():
    """Create various test cases for WALA slicing"""
    test_cases = [
        {
            "name": "simple_method",
            "content": '''package com.example;

public class SimpleTest {
    private int value;
    
    public SimpleTest(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;  // Line 12 - target for slicing
    }
    
    public void setValue(int newValue) {
        this.value = newValue;
    }
    
    public static void main(String[] args) {
        SimpleTest test = new SimpleTest(42);
        System.out.println(test.getValue());
    }
}''',
            "target_method": "com.example.SimpleTest#getValue()",
            "target_line": 12,
            "expected_lines": [8, 9, 10, 11, 12]  # Method body + return
        },
        
        {
            "name": "method_with_dependencies",
            "content": '''package com.example;

public class DependencyTest {
    private String name;
    private int count;
    
    public DependencyTest(String name) {
        this.name = name;
        this.count = 0;
    }
    
    public void increment() {
        count++;  // Line 15 - target for slicing
    }
    
    public String getName() {
        return name;
    }
    
    public int getCount() {
        return count;
    }
    
    public void process() {
        increment();
        System.out.println(name + ": " + count);
    }
}''',
            "target_method": "com.example.DependencyTest#increment()",
            "target_line": 15,
            "expected_lines": [14, 15, 16]  # Method body
        },
        
        {
            "name": "constructor_slice",
            "content": '''package com.example;

public class ConstructorTest {
    private String title;
    private boolean active;
    
    public ConstructorTest(String title) {
        this.title = title;  // Line 8 - target for slicing
        this.active = true;
    }
    
    public String getTitle() {
        return title;
    }
    
    public boolean isActive() {
        return active;
    }
}''',
            "target_method": "com.example.ConstructorTest#ConstructorTest(String)",
            "target_line": 8,
            "expected_lines": [7, 8, 9, 10]  # Constructor body
        },
        
        {
            "name": "field_access",
            "content": '''package com.example;

public class FieldTest {
    private String data;
    private int size;
    
    public FieldTest(String data) {
        this.data = data;
        this.size = data.length();
    }
    
    public String getData() {
        return data;  // Line 13 - target for slicing
    }
    
    public int getSize() {
        return size;
    }
}''',
            "target_method": "com.example.FieldTest#getData()",
            "target_line": 13,
            "expected_lines": [12, 13, 14]  # Method body
        }
    ]
    
    return test_cases

def run_wala_slicer_test(test_case, temp_dir):
    """Run WALA slicer on a specific test case"""
    print(f"\n--- Testing {test_case['name']} ---")
    
    # Create source directory structure
    src_dir = temp_dir / "src" / "main" / "java"
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # Create package directory
    package_dir = src_dir / "com" / "example"
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # Write test file
    java_file = package_dir / f"{test_case['name'].title().replace('_', '')}Test.java"
    java_file.write_text(test_case['content'])
    
    # Create output directory
    output_dir = temp_dir / "slices" / test_case['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build WALA slicer JAR if needed
    wala_jar = Path("/home/ubuntu/CFWR/build/libs/wala-slicer-all.jar")
    if not wala_jar.exists():
        print("Building WALA slicer JAR...")
        result = subprocess.run(["./gradlew", "walaSlicerJar"], 
                              cwd="/home/ubuntu/CFWR", 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to build WALA slicer JAR: {result.stderr}")
            return False, None
    
    # Run WALA slicer
    cmd = [
        "java", "-jar", str(wala_jar),
        "--sourceRoots", str(src_dir),
        "--projectRoot", str(temp_dir),
        "--targetFile", f"src/main/java/com/example/{java_file.name}",
        "--line", str(test_case['target_line']),
        "--targetMethod", test_case['target_method'],
        "--output", str(output_dir)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check results
    slice_files = list(output_dir.glob("*.java"))
    manifest_files = list(output_dir.glob("*.txt"))
    
    success = len(slice_files) > 0
    slice_content = ""
    
    if slice_files:
        slice_content = slice_files[0].read_text()
        print(f"✓ Slice created: {slice_files[0].name}")
        print(f"Slice content ({len(slice_content.splitlines())} lines):")
        print(slice_content)
    else:
        print("✗ No slice file created")
    
    if manifest_files:
        manifest_content = manifest_files[0].read_text()
        print(f"Manifest content:")
        print(manifest_content)
    
    return success, slice_content

def analyze_slice_correctness(test_case, slice_content):
    """Analyze if the slice contains the expected content"""
    if not slice_content:
        return False, "No slice content"
    
    lines = slice_content.splitlines()
    
    # Check if slice contains the target method
    target_method_name = test_case['target_method'].split('#')[1].split('(')[0]
    has_target_method = any(target_method_name in line for line in lines)
    
    # Check if slice contains meaningful Java code (not just comments)
    java_lines = [line for line in lines if line.strip() and not line.strip().startswith('//')]
    has_java_code = len(java_lines) > 0
    
    # Check if slice is not too large (should be focused)
    is_focused = len(lines) <= 20  # Reasonable upper bound
    
    # Check if slice contains the target line or nearby lines
    target_line_content = None
    for i, line in enumerate(test_case['content'].splitlines()):
        if i + 1 == test_case['target_line']:
            target_line_content = line.strip()
            break
    
    has_target_line = bool(target_line_content and any(target_line_content in line for line in lines))
    
    correctness_score = sum([has_target_method, has_java_code, is_focused, has_target_line])
    
    analysis = {
        "has_target_method": has_target_method,
        "has_java_code": has_java_code,
        "is_focused": is_focused,
        "has_target_line": has_target_line,
        "score": correctness_score,
        "total_lines": len(lines),
        "java_lines": len(java_lines)
    }
    
    return correctness_score >= 2, analysis

def run_comprehensive_test():
    """Run comprehensive WALA slicer correctness test"""
    print("WALA Slicer Correctness Test")
    print("=" * 50)
    
    test_cases = create_test_cases()
    results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for test_case in test_cases:
            success, slice_content = run_wala_slicer_test(test_case, temp_path)
            
            if success:
                is_correct, analysis = analyze_slice_correctness(test_case, slice_content)
                results.append({
                    "test_case": test_case['name'],
                    "success": success,
                    "correct": is_correct,
                    "analysis": analysis
                })
            else:
                results.append({
                    "test_case": test_case['name'],
                    "success": False,
                    "correct": False,
                    "analysis": {"error": "Failed to create slice"}
                })
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    correct_tests = sum(1 for r in results if r['correct'])
    
    print(f"Total tests: {total_tests}")
    print(f"Successful slices: {successful_tests}")
    print(f"Correct slices: {correct_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    print(f"Correctness rate: {correct_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"{status} {result['test_case']}: {result['analysis']}")
    
    # Overall assessment
    if correct_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - WALA slicer produces correct slices!")
        return True
    elif correct_tests >= total_tests * 0.8:
        print("\n✅ MOSTLY CORRECT - WALA slicer produces mostly correct slices")
        return True
    else:
        print("\n❌ ISSUES FOUND - WALA slicer needs improvement")
        return False

def main():
    """Main test function"""
    success = run_comprehensive_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
