#!/usr/bin/env python3
"""
Test script for the fixed WALA slicer implementation.
This script tests the WALA slicer with a simple Java file to ensure it produces non-blank slices.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

def create_test_java_file(test_dir):
    """Create a simple test Java file for WALA slicing"""
    java_content = '''package com.example;

public class TestClass {
    private int value;
    
    public TestClass(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;
    }
    
    public void setValue(int newValue) {
        this.value = newValue;
    }
    
    public void processData(int[] data) {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] * 2;
        }
    }
    
    public static void main(String[] args) {
        TestClass test = new TestClass(42);
        int[] numbers = {1, 2, 3, 4, 5};
        test.processData(numbers);
        System.out.println("Value: " + test.getValue());
    }
}
'''
    
    java_file = test_dir / "TestClass.java"
    java_file.write_text(java_content)
    return java_file

def test_wala_slicer():
    """Test the WALA slicer with a simple Java file"""
    print("Testing WALA slicer fixes...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test Java file
        java_file = create_test_java_file(temp_path)
        print(f"Created test file: {java_file}")
        
        # Create source roots directory
        src_dir = temp_path / "src" / "main" / "java"
        src_dir.mkdir(parents=True)
        
        # Move Java file to proper location
        package_dir = src_dir / "com" / "example"
        package_dir.mkdir(parents=True)
        target_file = package_dir / "TestClass.java"
        target_file.write_text(java_file.read_text())
        
        # Create output directory
        output_dir = temp_path / "slices"
        output_dir.mkdir()
        
        # Build WALA slicer JAR if it doesn't exist
        wala_jar = Path("/home/ubuntu/CFWR/build/libs/wala-slicer-all.jar")
        if not wala_jar.exists():
            print("Building WALA slicer JAR...")
            result = subprocess.run(["./gradlew", "walaSlicerJar"], 
                                  cwd="/home/ubuntu/CFWR", 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to build WALA slicer JAR: {result.stderr}")
                return False
        
        # Test WALA slicer
        cmd = [
            "java", "-jar", str(wala_jar),
            "--sourceRoots", str(src_dir),
            "--projectRoot", str(temp_path),
            "--targetFile", "src/main/java/com/example/TestClass.java",
            "--line", "15",  # Line with "return value;"
            "--targetMethod", "com.example.TestClass#getValue()",
            "--output", str(output_dir)
        ]
        
        print(f"Running WALA slicer: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        
        # Check if slice was created
        slice_files = list(output_dir.glob("*.java"))
        manifest_files = list(output_dir.glob("*.txt"))
        
        if slice_files:
            print(f"✓ Slice file created: {slice_files[0]}")
            slice_content = slice_files[0].read_text()
            print(f"Slice content ({len(slice_content.splitlines())} lines):")
            print(slice_content)
            return True
        else:
            print("✗ No slice file created")
            return False

def main():
    """Main test function"""
    print("WALA Slicer Test")
    print("=" * 50)
    
    success = test_wala_slicer()
    
    if success:
        print("\n✓ WALA slicer test PASSED")
        return 0
    else:
        print("\n✗ WALA slicer test FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
