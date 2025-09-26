# CFWR Annotation Placement System

This document provides a guide to the CFWR annotation placement system, which places relevant annotations on predicted locations for Java projects with support for the Lower Bound Checker's multiple annotations.

## Overview

The annotation placement system consists of three main components:

1. **`place_annotations.py`** - Core annotation placement script
2. **`predict_and_annotate.py`** - Integrated prediction and annotation pipeline  
3. **Enhanced annotation support** - Extended annotation types including Lower Bound Checker

## Features

### **Comprehensive Annotation Support**
- **Nullness Checker**: `@NonNull`, `@Nullable`, `@PolyNull`, `@MonotonicNonNull`
- **Index Checker**: `@IndexFor`, `@IndexOrLow`, `@IndexOrHigh`, `@LowerBound`, `@UpperBound`
- **Lower Bound Checker**: `@MinLen`, `@ArrayLen`, `@LTEqLengthOf`, `@GTLengthOf`, `@LengthOf`, `@Positive`, `@NonNegative`
- **Additional Annotations**: `@SameLen`, `@CapacityFor`, `@HasSubsequence`

### **Intelligent Placement Strategies**
- **Variable Declaration**: Places annotations before variable declarations
- **Method Parameters**: Annotates method parameters inline
- **Method Return Types**: Annotates return types
- **Field Declarations**: Annotates class fields
- **Array Access**: Special handling for array-related annotations
- **Loop Variables**: Context-aware loop variable annotations

### **Multiple Annotation Support**
- Places multiple annotations at the same location when appropriate
- Avoids duplicate annotations
- Context-aware annotation selection

### **Pipeline Integration**
- Seamlessly integrates with existing CFWR prediction pipeline
- Uses dataflow-augmented CFGs for better predictions
- Supports all three model types (HGT, GBT, Causal)

## Quick Start

### **1. Basic Annotation Placement**

```bash
# Place annotations based on existing predictions
python place_annotations.py \
  --project_root /path/to/java/project \
  --predictions_file predictions.json \
  --output_dir /path/to/output
```

### **2. Integrated Pipeline (Recommended)**

```bash
# Complete prediction and annotation pipeline
python predict_and_annotate.py \
  --project_root /path/to/java/project \
  --output_dir /path/to/output \
  --models hgt gbt causal
```

### **3. Test the System**

```bash
# Run comprehensive tests
python test_annotation_placement.py
```

## File Structure

```
CFWR/
├── place_annotations.py          # Core annotation placement
├── predict_and_annotate.py       # Integrated pipeline
├── test_annotation_placement.py  # Testing suite
├── annotation_placement.py       # Enhanced with Lower Bound support
└── ANNOTATION_PLACEMENT_GUIDE.md # This documentation
```

## Detailed Usage

### **Core Annotation Placement Script**

#### **Command Line Options**

```bash
python place_annotations.py [OPTIONS]

Required Arguments:
  --project_root PROJECT_ROOT    Root directory of the Java project
  --predictions_file PREDICTIONS_FILE  JSON file containing prediction results
  --output_dir OUTPUT_DIR        Output directory for processed files and reports

Optional Arguments:
  --backup                       Create backup of original files (default: True)
  --validate                     Validate annotations using Checker Framework (default: True)
  --checker_types {nullness,index,interning,lock,regex,signature}
                                Checker Framework checkers to use for validation
```

#### **Prediction File Format**

The predictions file should be in JSON format:

```json
[
  {
    "file_path": "com/example/MyClass.java",
    "line_number": 15,
    "confidence": 0.85,
    "annotation_type": "@NonNull",
    "target_element": "userName",
    "context": "String parameter in method",
    "model_type": "hgt"
  },
  {
    "file_path": "com/example/MyClass.java", 
    "line_number": 23,
    "confidence": 0.92,
    "annotation_type": "@MinLen",
    "target_element": "password",
    "context": "String field with length constraint",
    "model_type": "gbt"
  }
]
```

### **Integrated Pipeline Script**

#### **Command Line Options**

```bash
python predict_and_annotate.py [OPTIONS]

Required Arguments:
  --project_root PROJECT_ROOT    Root directory of the Java project to analyze
  --output_dir OUTPUT_DIR        Output directory for all results

Optional Arguments:
  --models_dir MODELS_DIR        Directory containing trained models
  --models {hgt,gbt,causal}      Models to use for prediction (default: all)
  --slicer {specimin,wala,cf}    Slicer to use (default: cf)
  --use_augmented_slices         Use augmented slices (default: True)
  --use_original_slices          Use original slices instead of augmented
  --dataflow_cfgs                Use dataflow-augmented CFGs (default: True)
  --validate_annotations         Validate placed annotations (default: True)
  --skip_validation              Skip annotation validation
```

#### **Output Structure**

```
output_dir/
├── predictions/
│   ├── hgt_predictions_dataflow.json
│   ├── gbt_predictions_dataflow.json
│   ├── causal_predictions_dataflow.json
│   └── merged_predictions.json
├── annotated_project/
│   ├── annotated_source/          # Modified project with annotations
│   ├── backups/                   # Original file backups
│   └── annotation_placement_report.md
├── temp_pipeline/                 # Temporary pipeline files
└── pipeline_summary.json         # Overall pipeline summary
```

## 🧠 **Annotation Selection Logic**

### **Context-Aware Selection**

The system analyzes code context to select appropriate annotations:

#### **Array Variables**
```java
// Input
int[] numbers = new int[size];

// Output
@MinLen(0)
int[] numbers = new int[size];
```

#### **Loop Variables**
```java
// Input
for (int i = 0; i < array.length; i++) {

// Output  
@NonNegative
@LTLengthOf("#1")
for (int i = 0; i < array.length; i++) {
```

#### **Index Variables**
```java
// Input
int index = findIndex(item);

// Output
@NonNegative
@IndexFor("#1")
int index = findIndex(item);
```

#### **Method Parameters**
```java
// Input
public void process(String data) {

// Output
public void process(@NonNull String data) {
```

### **Multiple Annotation Support**

The system can place multiple complementary annotations:

```java
// Input
String[] items = getItems();

// Output
@NonNull
@MinLen(1)
String[] items = getItems();
```

## Validation and Reporting

### **Automatic Validation**

The system can automatically validate placed annotations using the Checker Framework:

```bash
# Enable validation (default)
python place_annotations.py --validate

# Skip validation
python place_annotations.py --skip_validation
```

### **Generated Reports**

#### **Annotation Placement Report**

```markdown
# Annotation Placement Report

## Summary
- Total predictions processed: 25
- Successful placements: 23
- Failed placements: 2
- Skipped predictions: 0
- Success rate: 92.0%

## Files Processed

### com/example/UserService.java
- Annotations placed: 8
  - @NonNull: 4
  - @MinLen: 2
  - @Nullable: 2

### com/example/DataProcessor.java
- Annotations placed: 15
  - @NonNull: 6
  - @IndexFor: 4
  - @ArrayLen: 3
  - @Positive: 2

## Validation Results

### com/example/UserService.java
- Warnings after annotation: 2
- Annotations placed: 8

### com/example/DataProcessor.java  
- Warnings after annotation: 0
- Annotations placed: 15
```

#### **Pipeline Summary**

```json
{
  "pipeline_status": "completed",
  "prediction_files": {
    "hgt": "/path/to/hgt_predictions_dataflow.json",
    "gbt": "/path/to/gbt_predictions_dataflow.json", 
    "causal": "/path/to/causal_predictions_dataflow.json"
  },
  "merged_predictions": "/path/to/merged_predictions.json",
  "annotation_results": {
    "stats": {
      "total": 25,
      "successful": 23,
      "failed": 2,
      "skipped": 0
    },
    "annotated_project_dir": "/path/to/annotated_source",
    "report_path": "/path/to/annotation_placement_report.md"
  },
  "output_directory": "/path/to/output"
}
```

## Architecture

### **Core Classes**

#### **ComprehensiveAnnotationPlacer**
- Main orchestrator for annotation placement
- Handles multiple files and prediction formats
- Integrates with validation and reporting

#### **AnnotationContext**
- Analyzes code context around prediction locations
- Determines appropriate placement strategies
- Identifies variable types, method signatures, etc.

#### **IntegratedPipeline**
- Combines prediction and annotation placement
- Manages the complete end-to-end workflow
- Handles model ensemble and result merging

### **Placement Strategies**

1. **Variable Declaration**: `@NonNull String name;`
2. **Method Parameter**: `public void method(@NonNull String param)`
3. **Method Return**: `public @NonNull String getName()`
4. **Field Declaration**: `private @Nullable String field;`
5. **Array Access**: `@MinLen(0) int[] array;`
6. **Loop Variable**: `@NonNegative @LTLengthOf("#1") for (int i = ...)`

## Lower Bound Checker Integration

### **Supported Annotations**

| Annotation | Purpose | Example Usage |
|------------|---------|---------------|
| `@MinLen(value)` | Minimum array/string length | `@MinLen(1) String[] args` |
| `@ArrayLen(value)` | Exact array length | `@ArrayLen(3) int[] rgb` |
| `@LTEqLengthOf(value)` | Less than or equal to length | `@LTEqLengthOf("array") int index` |
| `@GTLengthOf(value)` | Greater than length | `@GTLengthOf("array") int capacity` |
| `@LengthOf(value)` | Equal to length | `@LengthOf("array") int size` |
| `@Positive` | Greater than zero | `@Positive int count` |
| `@NonNegative` | Greater than or equal to zero | `@NonNegative int index` |

### **Context-Aware Placement**

```java
// Array declaration
@MinLen(0)
int[] numbers = new int[size];

// Index variable
@NonNegative
@IndexFor("numbers")  
int currentIndex = 0;

// Length parameter
public void resize(@Positive int newSize) {
    // implementation
}

// Capacity field
@GTLengthOf("data")
private int capacity;
```

## Testing

### **Test Categories**

1. **Basic Functionality Tests**
   - Annotation placement accuracy
   - Multiple annotation handling
   - File backup and restoration

2. **Integration Tests**
   - End-to-end pipeline execution
   - Model ensemble integration
   - Validation system integration

3. **Edge Case Tests**
   - Empty files
   - Malformed predictions
   - Invalid file paths

### **Running Tests**

```bash
# Run all tests
python test_annotation_placement.py

# Test with specific project
python predict_and_annotate.py --project_root /path/to/test/project --output_dir /tmp/test_output
```

## 📈 **Performance Considerations**

### **Optimization Strategies**

1. **File Caching**: AnnotationPlacementManagers are cached per file
2. **Batch Processing**: Multiple predictions per file are processed together
3. **Reverse Order Processing**: Lines processed in reverse to avoid line shift issues
4. **Lazy Validation**: Validation only runs when requested

### **Scalability**

- Handles projects with thousands of files
- Processes hundreds of predictions efficiently  
- Memory-efficient file processing
- Parallel model execution support

## Customization

### **Adding New Annotation Types**

1. **Extend AnnotationType enum** in `annotation_placement.py`:
```python
class AnnotationType(Enum):
    # ... existing annotations ...
    MY_CUSTOM_ANNOTATION = "@MyCustom"
```

2. **Update annotation selection logic** in `place_annotations.py`:
```python
def select_appropriate_annotation(self, prediction, context):
    # ... existing logic ...
    if 'custom_context' in context.code_line:
        annotations.append('@MyCustom')
```

### **Custom Placement Strategies**

```python
def _format_annotations_for_strategy(self, annotations, strategy, lines, line_number):
    if strategy == AnnotationStrategy.MY_CUSTOM_STRATEGY:
        # Custom formatting logic
        return insert_line, annotation_text
    # ... existing logic ...
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **No Annotations Placed**
- Check prediction file format
- Verify file paths are correct
- Ensure target files exist and are readable

#### **Validation Errors**
- Check Checker Framework installation
- Verify classpath configuration
- Ensure Java source compatibility

#### **Performance Issues**
- Use smaller batch sizes for large projects
- Skip validation for initial testing
- Process files in parallel (future enhancement)

### **Debug Mode**

```bash
# Enable debug logging
export CFWR_DEBUG=1
python place_annotations.py --project_root /path/to/project --predictions_file predictions.json --output_dir /path/to/output
```

## Future Enhancements

### **Planned Features**

1. **Interactive Mode**: Manual review of annotation placements
2. **Confidence Thresholding**: Only place high-confidence predictions  
3. **Batch Validation**: Parallel Checker Framework execution
4. **IDE Integration**: Direct integration with development environments
5. **Annotation Refinement**: Machine learning-based annotation improvement

### **Advanced Strategies**

1. **Contextual Annotation Chaining**: Related annotations across methods
2. **Type Inference Integration**: Leverage Java type system
3. **Documentation Generation**: Automatic documentation of placed annotations
4. **Annotation Clustering**: Group related annotations for better organization

## 📚 **References**

- [Checker Framework Manual](https://checkerframework.org/manual/)
- [Lower Bound Checker Documentation](https://checkerframework.org/manual/#lower-bound-checker)
- [Index Checker Documentation](https://checkerframework.org/manual/#index-checker)
- [Nullness Checker Documentation](https://checkerframework.org/manual/#nullness-checker)

---

**Note**: This annotation placement system represents a significant advancement in automated Java annotation placement, providing comprehensive support for multiple Checker Framework annotation types with intelligent, context-aware placement strategies.
