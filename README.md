# Checker Framework Warning Resolver (CFWR)

A comprehensive machine learning pipeline for predicting Checker Framework annotation placements in Java code. The system uses Checker Framework warnings to generate code slices, converts them to dataflow-augmented Control Flow Graphs (CFGs), and trains multiple ML models to predict where annotations should be placed.

## 🎯 **Overview**

CFWR implements a complete end-to-end pipeline that:

1. **Analyzes** Checker Framework warnings from Java projects
2. **Slices** code using Checker Framework slicer (default), Specimin, or WALA slicers
3. **Augments** slices with syntactically correct but irrelevant code (default behavior)
4. **Converts** slices to dataflow-augmented Control Flow Graphs (CFGs) for structural analysis
5. **Trains** three different ML models: HGT, GBT, and Causal models
6. **Predicts** annotation placements using AST-based analysis
7. **Places** annotations exactly where they should be, not approximately

## ✨ **Key Features**

### **🔧 Best Practices Defaults**
- **Augmented slices by default**: Training scripts automatically use augmented slices
- **Dataflow-augmented CFGs**: CFGs include dataflow information by default
- **Checker Framework slicer**: Uses CF slicer as default for better quality

### **🚀 Comprehensive Support**
- **All Checker Framework annotations**: Including Lower Bound Checker annotations
- **Multiple slicers**: Checker Framework (default), Specimin, WALA
- **Three ML models**: HGT, GBT, and Causal models
- **Flexible prediction**: Individual files, directories, or entire projects
- **Reinforcement Learning**: Advanced RL training with Checker Framework feedback

## 🏗️ **Architecture**

### **Core Components**

```
CFWR Pipeline
├── Slicing (Checker Framework slicer - default)
├── Data Augmentation (automatic - default)
├── CFG Generation (with dataflow - default)
├── Model Training (HGT, GBT, Causal)
├── Annotation Placement (AST-based - default)
└── Validation (Checker Framework integration)
```

### **Placement System**

- **`perfect_annotation_placement.py`**: Core placement engine
- **`place_annotations.py`**: Main annotation placement script
- **`predict_and_annotate.py`**: Integrated prediction and placement pipeline
- **AST-based analysis**: Precise code structure understanding
- **Context-aware strategies**: Intelligent placement based on code context

## 🚀 **Quick Start**

### **Prerequisites**

- Java 21+
- Python 3.8+
- Gradle 7+
- Checker Framework 3.42.0+

### **Installation**

1. **Clone and initialize**:
   ```bash
   git clone <repository-url>
   cd CFWR
   git submodule update --init --recursive
   ```

2. **Set up Checker Framework**:
   ```bash
   export CHECKERFRAMEWORK_HOME="/path/to/checker-framework-3.42.0"
   export CHECKERFRAMEWORK_CP="/path/to/checker-framework-3.42.0/checker/dist/checker-qual.jar:/path/to/checker-framework-3.42.0/checker/dist/checker.jar"
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Build the project**:
   ```bash
   ./gradlew build
   ```

## 📖 **Usage**

### **🎯 Complete Integrated Pipeline (Recommended)**

The easiest way to use CFWR is through the integrated pipeline that handles everything automatically:

```bash
# Complete pipeline: slice → augment → train → predict → place annotations
python predict_and_annotate.py \
  --project_root /path/to/java/project \
  --output_dir /path/to/output \
  --models hgt gbt causal
```

This single command:
- ✅ Generates slices using Checker Framework slicer (default)
- ✅ Augments slices automatically (default behavior)
- ✅ Creates dataflow-augmented CFGs (default)
- ✅ Trains all three models
- ✅ Places annotations (default)

### **🔧 Individual Components**

#### **1. Annotation Placement Only**

```bash
# Place annotations based on existing predictions
python place_annotations.py \
  --project_root /path/to/java/project \
  --predictions_file predictions.json \
  --output_dir /path/to/output
```

#### **2. Project-Wide Prediction**

```bash
# Generate predictions for an entire project
python predict_on_project.py \
  --project_root /path/to/java/project \
  --output_dir /path/to/output \
  --models hgt gbt causal
```

#### **3. Individual Model Training**

```bash
# Train individual models (uses augmented slices by default)
python hgt.py      # Heterogeneous Graph Transformer
python gbt.py      # Gradient Boosting Trees  
python causal_model.py  # Causal inference model
```

#### **4. Individual Model Prediction**

```bash
# HGT predictions
python predict_hgt.py \
  --slices_dir slices_aug \
  --model_path models/hgt_model.pth \
  --out_path predictions_hgt.json

# GBT predictions
python predict_gbt.py \
  --slices_dir slices_aug \
  --model_path models/gbt_model.joblib \
  --out_path predictions_gbt.json

# Causal model predictions
python predict_causal.py \
  --slices_dir slices_aug \
  --model_path models/causal_model.joblib \
  --out_path predictions_causal.json
```

### **🎛️ Advanced Options**

#### **Perfect vs Approximate Placement**

```bash
# Perfect placement (DEFAULT - recommended)
python place_annotations.py \
  --project_root /path/to/project \
  --predictions_file predictions.json \
  --output_dir /path/to/output \
  --perfect_placement

# Approximate placement (BACKUP - less accurate)
python place_annotations.py \
  --project_root /path/to/project \
  --predictions_file predictions.json \
  --output_dir /path/to/output \
  --approximate_placement
```

#### **Slicer Selection**

```bash
# Checker Framework slicer (DEFAULT - recommended)
python predict_and_annotate.py \
  --project_root /path/to/project \
  --output_dir /path/to/output \
  --slicer cf

# Specimin slicer
python predict_and_annotate.py \
  --project_root /path/to/project \
  --output_dir /path/to/output \
  --slicer specimin

# WALA slicer
python predict_and_annotate.py \
  --project_root /path/to/project \
  --output_dir /path/to/output \
  --slicer wala
```

#### **Slice Type Selection**

```bash
# Use augmented slices (DEFAULT - recommended)
python predict_and_annotate.py \
  --project_root /path/to/project \
  --output_dir /path/to/output \
  --use_augmented_slices

# Use original slices
python predict_and_annotate.py \
  --project_root /path/to/project \
  --output_dir /path/to/output \
  --use_original_slices
```

## 🧠 **Model Details**

### **HGT (Heterogeneous Graph Transformer)**
- **Type**: Graph neural network
- **Input**: Dataflow-augmented Control Flow Graphs as heterogeneous graphs
- **Output**: Node-level predictions for annotation placement
- **Best for**: Complex control flow patterns and structural relationships
- **Features**: Uses dataflow edges for better graph representation

### **GBT (Gradient Boosting Trees)**
- **Type**: Ensemble learning
- **Input**: CFG-level features including dataflow information
- **Output**: CFG-level predictions applied to all nodes
- **Best for**: Fast predictions and interpretable feature importance
- **Features**: Enhanced with dataflow features (dataflow_count, control_count, etc.)

### **Causal Model**
- **Type**: Predictive classifier (simplified from causal inference)
- **Input**: Node features with dataflow relationships
- **Output**: Node-level predictions based on feature analysis
- **Best for**: Understanding annotation patterns and feature importance
- **Features**: Includes dataflow complexity metrics

## 🎨 **Annotation Support**

### **Supported Annotation Types**

#### **Nullness Checker**
- `@NonNull`, `@Nullable`, `@PolyNull`, `@MonotonicNonNull`

#### **Index Checker**
- `@IndexFor`, `@IndexOrLow`, `@IndexOrHigh`, `@LowerBound`, `@UpperBound`

#### **Lower Bound Checker** (Enhanced Support)
- `@MinLen`, `@ArrayLen`, `@LTEqLengthOf`, `@GTLengthOf`, `@LengthOf`
- `@Positive`, `@NonNegative`, `@GTENegativeOne`, `@LTLengthOf`
- `@SearchIndexFor`, `@SearchIndexBottom`, `@SearchIndexUnknown`

#### **Additional Annotations**
- `@SameLen`, `@CapacityFor`, `@HasSubsequence`

### **Perfect Placement Examples**

```java
// Field Declaration
@NonNull private String name;

// Method Parameter
public void method(@NonNull String param) {

// Constructor Parameter
public Test(@NonNull String name) {

// Multiple Annotations
public ComplexTest(@NonNull @MinLen(0) String name, int[] numbers) {

// Array Field
@MinLen(0) private int[] numbers;
```

## 🔄 **Data Augmentation**

The system includes sophisticated data augmentation that:

- **Preserves semantics**: Keeps original code logic intact
- **Adds variety**: Introduces syntactically correct but irrelevant code
- **Increases robustness**: Helps models generalize to diverse code patterns
- **Maintains structure**: Preserves CFG properties for graph-based models
- **Default behavior**: Automatically used by all training scripts

### **Augmentation Methods**
- Random method insertion
- Variable declaration injection
- Control flow statement addition
- Expression complexity variation
- Dynamic code generation (not just templates)

## 🧪 **Testing**

### **Run All Tests**

```bash
# Test perfect annotation placement accuracy
python test_perfect_accuracy.py

# Test default behavior (perfect placement)
python test_default_perfect_placement.py

# Test annotation placement system
python test_annotation_placement.py

# Test all model types
python test_all_models.py
```

## 📁 **File Structure**

```
CFWR/
├── Core Scripts
│   ├── place_annotations.py              # Main annotation placement (perfect accuracy)
│   ├── predict_and_annotate.py          # Integrated pipeline
│   ├── predict_on_project.py            # Project-wide prediction
│   └── perfect_annotation_placement.py  # Perfect placement engine
│
├── Training Scripts
│   ├── hgt.py                           # HGT training (augmented slices default)
│   ├── gbt.py                           # GBT training (augmented slices default)
│   ├── causal_model.py                  # Causal model training (augmented slices default)
│   └── enhanced_rl_training.py          # Reinforcement learning training
│
├── Prediction Scripts
│   ├── predict_hgt.py                   # HGT predictions
│   ├── predict_gbt.py                   # GBT predictions
│   └── predict_causal.py                # Causal model predictions
│
├── Pipeline Scripts
│   ├── pipeline.py                       # Main pipeline orchestration
│   ├── augment_slices.py                # Data augmentation (dynamic generation)
│   ├── cfg.py                           # CFG generation (with dataflow)
│   └── rl_pipeline.py                   # RL training pipeline
│
├── Integration Modules
│   ├── annotation_placement.py           # Enhanced annotation placement
│   ├── checker_framework_integration.py # Checker Framework integration
│   └── annotation_placement.py          # Annotation placement utilities
│
├── Java Components
│   ├── src/main/java/cfwr/              # Java resolver and slicer integration
│   └── build.gradle                     # Gradle build configuration
│
├── Documentation
│   ├── README.md                        # This file
│   ├── ANNOTATION_PLACEMENT_GUIDE.md    # Comprehensive placement guide
│   ├── BEST_PRACTICES_DEFAULTS.md       # Best practices documentation
│   └── AUGMENTED_SLICES_DEFAULT.md      # Augmented slices documentation
│
├── Test Scripts
│   ├── test_perfect_accuracy.py         # Perfect placement accuracy tests
│   ├── test_default_perfect_placement.py # Default behavior tests
│   ├── test_annotation_placement.py     # Annotation placement tests
│   └── test_all_models.py               # All model type tests
│
└── Generated Directories
    ├── models/                          # Trained model files
    ├── slices/                          # Original code slices
    ├── slices_aug/                      # Augmented slices (auto-generated)
    ├── cfg_output/                      # Generated CFGs (with dataflow)
    └── predictions/                    # Prediction results
```

## 🌟 **Examples**

### **Complete Workflow Example**

```bash
# 1. Complete integrated pipeline (recommended)
python predict_and_annotate.py \
  --project_root /home/user/project \
  --output_dir /home/user/results \
  --models hgt gbt causal

# This automatically:
# ✅ Generates slices using Checker Framework slicer
# ✅ Augments slices for better training
# ✅ Creates dataflow-augmented CFGs
# ✅ Trains all three models
# ✅ Places annotations with perfect accuracy
# ✅ Validates results with Checker Framework
```

### **Step-by-Step Example**

```bash
# 1. Generate slices and CFGs
python pipeline.py \
  --project_root /home/user/project \
  --warnings_file warnings.out \
  --output_dir results \
  --slicer cf

# 2. Train models (uses augmented slices by default)
python hgt.py
python gbt.py  
python causal_model.py

# 3. Place annotations with perfect accuracy
python place_annotations.py \
  --project_root /home/user/project \
  --predictions_file results/predictions.json \
  --output_dir results/annotated
```

### **Reinforcement Learning Example**

```bash
# Train models with RL feedback from Checker Framework
python enhanced_rl_training.py \
  --project_root /home/user/project \
  --output_dir results/rl_training \
  --models hgt gbt causal \
  --use_augmented_slices
```

## 🔧 **Environment Variables**

Configure the pipeline using environment variables:

```bash
export SLICES_DIR="/path/to/slices"                    # Slices directory
export CFG_OUTPUT_DIR="/path/to/cfg_output"           # CFG output directory  
export MODELS_DIR="/path/to/models"                    # Models directory
export CHECKERFRAMEWORK_CP="/path/to/checker-jars"    # Checker Framework classpath
export AUGMENTED_SLICES_DIR="/path/to/slices_aug"     # Augmented slices directory
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **"No slices directory found"**: Ensure slices are generated and SLICES_DIR is set correctly
2. **"Model not found"**: Train models first using the training scripts
3. **"ClassNotFoundException"**: Ensure Checker Framework JARs are in the classpath
4. **"Perfect placement failed"**: Check Java syntax and file permissions
5. **"Parsing warnings"**: Some augmented slices may have parsing issues - this is normal and handled gracefully

### **Debug Mode**

Enable verbose output for debugging:

```bash
export CFWR_DEBUG=1
python place_annotations.py  # Will show detailed progress information
```

### **Fallback Options**

If perfect placement fails, you can use approximate placement:

```bash
python place_annotations.py \
  --project_root /path/to/project \
  --predictions_file predictions.json \
  --output_dir /path/to/output \
  --approximate_placement
```
