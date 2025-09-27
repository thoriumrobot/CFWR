#!/usr/bin/env python3
"""
Integration test for WALA slicer with Checker Framework Warning Resolver.
This test verifies that the WALA slicer works correctly in the CFWR pipeline context.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

def create_test_warnings_file(temp_dir):
    """Create a test warnings file for CFWR"""
    warnings_content = '''/tmp/test/src/main/java/com/example/ArrayTest.java:15:17: compiler.err.proc.messager: [index] Possible out-of-bounds access
/tmp/test/src/main/java/com/example/ArrayTest.java:23:21: compiler.err.proc.messager: [index] Possible out-of-bounds access
'''
    
    warnings_file = temp_dir / "test_warnings.out"
    warnings_file.write_text(warnings_content)
    return warnings_file

def create_test_java_project(temp_dir):
    """Create a test Java project with Checker Framework warnings"""
    # Create project structure
    src_dir = temp_dir / "src" / "main" / "java" / "com" / "example"
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test Java file
    java_content = '''package com.example;

import org.checkerframework.checker.index.qual.*;

public class ArrayTest {
    public void testArrayAccess(int[] array, int index) {
        // This will generate a warning
        int value = array[index];  // Line 8 - warning location
        
        System.out.println("Value: " + value);
    }
    
    public void testLoop(int[] data) {
        for (int i = 0; i < data.length; i++) {
            // This will also generate a warning
            int item = data[i];  // Line 15 - warning location
            processItem(item);
        }
    }
    
    private void processItem(int item) {
        System.out.println("Processing: " + item);
    }
    
    public static void main(String[] args) {
        ArrayTest test = new ArrayTest();
        int[] numbers = {1, 2, 3, 4, 5};
        test.testArrayAccess(numbers, 2);
        test.testLoop(numbers);
    }
}'''
    
    java_file = src_dir / "ArrayTest.java"
    java_file.write_text(java_content)
    
    return temp_dir

def test_cfwr_integration():
    """Test WALA slicer integration with CFWR"""
    print("Testing WALA slicer integration with CFWR...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test project
        project_root = create_test_java_project(temp_path)
        
        # Create test warnings file
        warnings_file = create_test_warnings_file(temp_path)
        
        # Update warnings file with correct paths
        warnings_content = warnings_file.read_text()
        warnings_content = warnings_content.replace('/tmp/test', str(project_root))
        warnings_file.write_text(warnings_content)
        
        print(f"Project root: {project_root}")
        print(f"Warnings file: {warnings_file}")
        
        # Set environment variables for WALA slicer
        os.environ['SLICES_DIR'] = str(temp_path / "slices_wala")
        os.environ['CHECKERFRAMEWORK_CP'] = "/home/ubuntu/checker-framework-3.42.0/checker/dist/checker-qual.jar:/home/ubuntu/checker-framework-3.42.0/checker/dist/checker.jar"
        
        # Run CheckerFrameworkWarningResolver with WALA slicer
        cmd = [
            "java", "-cp", "/home/ubuntu/CFWR/build/libs/CFWR-all.jar",
            "cfwr.CheckerFrameworkWarningResolver",
            str(project_root),
            str(warnings_file),
            "/home/ubuntu/CFWR",
            "wala"
        ]
        
        print(f"Running CFWR with WALA slicer: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/ubuntu/CFWR")
        
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        
        # Check if slices were created
        slices_dir = temp_path / "slices_wala"
        if slices_dir.exists():
            slice_files = list(slices_dir.glob("**/*.java"))
            manifest_files = list(slices_dir.glob("**/*.txt"))
            
            print(f"\nSlices created: {len(slice_files)}")
            print(f"Manifests created: {len(manifest_files)}")
            
            if slice_files:
                print("\nSlice files:")
                for slice_file in slice_files:
                    print(f"  {slice_file}")
                    content = slice_file.read_text()
                    print(f"    Content ({len(content.splitlines())} lines):")
                    print(f"    {content[:200]}...")
            
            if manifest_files:
                print("\nManifest files:")
                for manifest_file in manifest_files:
                    print(f"  {manifest_file}")
                    content = manifest_file.read_text()
                    print(f"    Content:")
                    print(f"    {content}")
            
            return len(slice_files) > 0
        else:
            print("No slices directory created")
            return False

def main():
    """Main test function"""
    print("WALA Slicer Integration Test")
    print("=" * 50)
    
    success = test_cfwr_integration()
    
    if success:
        print("\n✅ WALA slicer integration test PASSED")
        print("The WALA slicer works correctly with the CFWR pipeline")
        return 0
    else:
        print("\n❌ WALA slicer integration test FAILED")
        print("The WALA slicer needs further integration work")
        return 1

if __name__ == "__main__":
    sys.exit(main())
