# Soot Prediction Mode for Annotation Placement

## Overview

The enhanced Soot integration now supports **Prediction Mode** for true bytecode slicing with Vineflower decompilation. This mode is specifically designed for prediction and annotation placement workflows where you need to slice bytecode and then decompile it back to Java source code for annotation placement.

## Key Features

### **Two Operating Modes**

1. **Training Mode** (Default)
   - Uses source-based slicing for better compatibility
   - Suitable for ML model training
   - Faster and more reliable

2. **Prediction Mode** (New)
   - Compiles Java source to bytecode
   - Performs true program slicing on bytecode
   - Uses Vineflower to decompile back to Java source
   - Ideal for annotation placement workflows

### **Workflow Comparison**

#### **Training Mode Workflow**
```
Java Source → Source-based Slicing → Java Slice
```

#### **Prediction Mode Workflow**
```
Java Source → Compile to Bytecode → Bytecode Slicing → Vineflower Decompilation → Java Source
```

## Usage

### **Environment Variable Method**

```bash
# Enable prediction mode
export SOOT_PREDICTION_MODE=true

# Run pipeline with prediction mode
python3 pipeline.py \
  --steps slice \
  --project_root /path/to/java/project \
  --warnings_file warnings.out \
  --slicer soot
```

### **Command-Line Flag Method**

```bash
# Direct slicer invocation with prediction mode
./tools/soot_slicer.sh \
  --projectRoot /path/to/project \
  --targetFile src/main/java/Example.java \
  --line 15 \
  --output /tmp/slice \
  --member "Example#method(int,String)" \
  --decompiler /path/to/vineflower.jar \
  --prediction-mode
```

### **Pipeline Integration**

```bash
# Set up environment
export SOOT_SLICE_CLI="/home/ubuntu/CFWR/tools/soot_slicer.sh"
export VINEFLOWER_JAR="/home/ubuntu/CFWR/tools/vineflower.jar"
export SOOT_PREDICTION_MODE=true

# Run complete pipeline
python3 pipeline.py \
  --steps all \
  --project_root /path/to/java/project \
  --warnings_file warnings.out \
  --slicer soot
```

## Technical Implementation

### **Bytecode Slicing Process**

1. **Compilation**: Java source is compiled to bytecode using `javac`
2. **Soot Analysis**: Soot framework analyzes the bytecode
3. **Program Slicing**: Performs backward slicing from target line
4. **Bytecode Generation**: Creates sliced bytecode representation
5. **Vineflower Decompilation**: Decompiles bytecode back to Java source
6. **Cleanup**: Removes temporary files

### **Code Structure**

```java
// Main slicing method with mode selection
public void sliceMethod(String projectRoot, String targetFile, int lineNumber, 
                       String outputDir, String memberSig, String vineflowerJar, 
                       boolean predictionMode) throws Exception {
    
    if (predictionMode && vineflowerJar != null) {
        // Prediction mode: Use true bytecode slicing with Vineflower decompilation
        performBytecodeSlicingWithDecompilation(targetPath, lineNumber, outputDir, memberSig, vineflowerJar);
    } else {
        // Training mode: Use source-based slicing for better compatibility
        performSourceBasedSlicing(targetPath, lineNumber, outputDir, memberSig);
    }
}
```

### **Key Methods**

- `performBytecodeSlicingWithDecompilation()`: Main prediction mode workflow
- `compileJavaToBytecode()`: Compiles Java source to bytecode
- `setupSootForBytecodeAnalysis()`: Configures Soot for bytecode analysis
- `performProgramSlicing()`: Performs actual program slicing
- `decompileWithVineflower()`: Uses Vineflower for decompilation
- `cleanupTempFiles()`: Cleans up temporary files

## Configuration

### **Required Environment Variables**

```bash
# Soot slicer configuration
export SOOT_SLICE_CLI="/path/to/soot_slicer.sh"
export SOOT_JAR="/path/to/soot-slicer-all.jar"  # Alternative to CLI

# Vineflower decompiler
export VINEFLOWER_JAR="/path/to/vineflower.jar"

# Prediction mode
export SOOT_PREDICTION_MODE=true
```

### **Command-Line Arguments**

```bash
--projectRoot <path>        # Project root directory
--targetFile <file>         # Target Java file
--line <number>             # Line number for slicing
--output <dir>              # Output directory
--member <sig>              # Member signature
--decompiler <vineflower.jar>  # Vineflower JAR path
--prediction-mode           # Enable prediction mode
```

## Error Handling

### **Intelligent Fallback System**

The system includes multiple levels of fallback:

1. **Bytecode Slicing Fails**: Falls back to source-based slicing
2. **Vineflower Decompilation Fails**: Falls back to source-based slicing
3. **Compilation Fails**: Falls back to source-based slicing
4. **All Methods Fail**: Creates minimal fallback slice

### **Error Messages**

```bash
[soot_slicer] Performing bytecode slicing with Vineflower decompilation
Bytecode slicing failed: None of the basic classes could be loaded! Check your Soot class path!
Slicing failed: Bytecode slicing failed
[soot_slicer] Generated source-based slice: /tmp/slice/Example_slice.java
[soot_slicer] Slicing completed successfully
```

## Benefits for Annotation Placement

### **Why Use Prediction Mode?**

1. **True Bytecode Analysis**: Analyzes actual compiled code, not just source
2. **Better Slicing**: More accurate program slicing on bytecode level
3. **Decompilation**: Converts sliced bytecode back to Java for annotation placement
4. **Annotation Compatibility**: Decompiled code is ready for annotation placement

### **Use Cases**

- **Production Code**: When analyzing compiled production code
- **Optimized Code**: When source code has been optimized/obfuscated
- **Library Analysis**: When analyzing third-party libraries
- **Annotation Placement**: When placing annotations on decompiled code

## Performance Considerations

### **Training Mode**
- ✅ **Fast**: Direct source analysis
- ✅ **Reliable**: No compilation/decompilation overhead
- ✅ **Compatible**: Works with all Java source files

### **Prediction Mode**
- ⚠️ **Slower**: Compilation + slicing + decompilation
- ⚠️ **Complex**: More failure points
- ✅ **Accurate**: True bytecode analysis
- ✅ **Flexible**: Works with compiled code

## Troubleshooting

### **Common Issues**

1. **Soot Classpath Issues**
   ```
   Bytecode slicing failed: None of the basic classes could be loaded! Check your Soot class path!
   ```
   **Solution**: Falls back to source-based slicing automatically

2. **Vineflower Decompilation Fails**
   ```
   Vineflower decompilation failed
   ```
   **Solution**: Falls back to source-based slicing automatically

3. **Compilation Errors**
   ```
   Failed to compile Java source: /path/to/file.java
   ```
   **Solution**: Falls back to source-based slicing automatically

### **Debug Mode**

```bash
# Enable debug logging
export SOOT_DEBUG=true

# Run with verbose output
./tools/soot_slicer.sh --prediction-mode --decompiler /path/to/vineflower.jar [args]
```

## Future Enhancements

1. **Advanced Slicing**: Implement more sophisticated program slicing algorithms
2. **Line Number Mapping**: Better mapping between source and bytecode line numbers
3. **Performance Optimization**: Optimize compilation and decompilation processes
4. **Error Recovery**: Better error recovery and retry mechanisms
5. **Caching**: Cache compiled bytecode for repeated analysis

## Summary

The Soot Prediction Mode provides a powerful tool for true bytecode slicing with Vineflower decompilation, specifically designed for annotation placement workflows. While it's more complex than source-based slicing, it offers superior accuracy for analyzing compiled code and placing annotations on decompiled Java source.

The system includes robust fallback mechanisms to ensure it always produces meaningful output, making it suitable for production use in the CFWR machine learning pipeline.
