# Checker Framework Warning Resolver (CFWR)

A machine learning pipeline for predicting Lower Bound Checker annotation placements in Java code. The system uses Checker Framework warnings to generate code slices, converts them to Control Flow Graphs (CFGs), and trains multiple ML models to predict where annotations should be placed.

## Overview

CFWR implements a complete pipeline that:
1. **Analyzes** Checker Framework warnings from Java projects
2. **Slices** code using either Specimin or WALA slicers to extract relevant code segments
3. **Augments** slices with syntactically correct but irrelevant code to increase training data diversity
4. **Converts** slices to Control Flow Graphs (CFGs) for structural analysis
5. **Trains** three different ML models: HGT (Heterogeneous Graph Transformer), GBT (Gradient Boosting Trees), and Causal models
6. **Predicts** annotation placements on new code using trained models

## Key Features

- **Automatic slicer selection**: Choose between Specimin (source-based) or WALA (bytecode-based) slicers
- **Data augmentation**: Automatically generates augmented training data with irrelevant code variations
- **Multiple ML models**: HGT for graph-based learning, GBT for feature-based learning, and Causal models for causal inference
- **Flexible prediction**: Support for both individual file and project-wide prediction
- **Best practices defaults**: Training scripts automatically use augmented slices when available

## Setup

### Prerequisites

- Java 21+
- Python 3.8+
- Gradle 7+
- Checker Framework 3.42.0+

### Installation

1. **Clone and initialize submodules**:
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

## Usage

### Basic Pipeline

#### 1. Generate Slices from Warnings

```bash
# Using Specimin slicer (recommended)
./gradlew runResolver -Pargs="/path/to/project /path/to/warnings.out /path/to/CFWR specimin"

# Using WALA slicer
./gradlew runResolver -Pargs="/path/to/project /path/to/warnings.out /path/to/CFWR wala"
```

#### 2. Augment Training Data (Optional but Recommended)

```bash
python3 augment_slices.py --slices_dir slices --output_dir slices_aug --variants 3
```

#### 3. Train Models

The training scripts automatically use augmented slices if available, falling back to regular slices:

```bash
# Train all models (automatically uses best available data)
python3 hgt.py      # Heterogeneous Graph Transformer
python3 gbt.py      # Gradient Boosting Trees  
python3 causal_model.py  # Causal inference model
```

#### 4. Generate Predictions

**Individual file prediction**:
```bash
# HGT predictions
python3 predict_hgt.py --java_file /path/to/slice.java \
                       --model_path models/best_model.pth \
                       --out_path predictions_hgt.json

# GBT predictions
python3 predict_gbt.py --java_file /path/to/slice.java \
                      --model_path models/gbt_model_1.joblib \
                      --out_path predictions_gbt.json

# Causal model predictions
python3 predict_causal.py --java_file /path/to/slice.java \
                         --model_path models/causal_clf.joblib \
                         --out_path predictions_causal.json
```

**Directory-based prediction** (for multiple slices):
```bash
# Predict on all slices in a directory
python3 predict_hgt.py --slices_dir slices_aug \
                      --model_path models/best_model.pth \
                      --out_path predictions_hgt.json
```

**Project-wide prediction** (comprehensive pipeline):
```bash
# Run complete pipeline on a target project
python3 predict_on_project.py --project_root /path/to/target/project \
                              --models_dir models \
                              --output_dir project_predictions \
                              --slicer specimin \
                              --models hgt gbt causal
```

### Advanced Usage

#### Environment Variables

Configure the pipeline using environment variables:

```bash
export SLICES_DIR="/path/to/slices"                    # Slices directory
export CFG_OUTPUT_DIR="/path/to/cfg_output"           # CFG output directory  
export MODELS_DIR="/path/to/models"                    # Models directory
export CHECKERFRAMEWORK_CP="/path/to/checker-jars"    # Checker Framework classpath
export SPECIMIN_JARPATH="/path/to/specimin/jar"       # Specimin JAR path
```

#### Pipeline Script

Use the comprehensive pipeline script for automated workflows:

```bash
# Complete pipeline: slice → augment → train → predict
python3 pipeline.py --project_root /path/to/project \
                   --warnings_file /path/to/warnings.out \
                   --output_dir results \
                   --slicer specimin \
                   --augment_variants 3 \
                   --models hgt gbt causal
```

## Model Details

### HGT (Heterogeneous Graph Transformer)
- **Type**: Graph neural network
- **Input**: Control Flow Graphs as heterogeneous graphs
- **Output**: Node-level predictions for annotation placement
- **Best for**: Complex control flow patterns and structural relationships

### GBT (Gradient Boosting Trees)
- **Type**: Ensemble learning
- **Input**: CFG-level features (complexity, node counts, etc.)
- **Output**: CFG-level predictions applied to all nodes
- **Best for**: Fast predictions and interpretable feature importance

### Causal Model
- **Type**: Causal inference classifier
- **Input**: Node features with causal relationships
- **Output**: Node-level predictions based on causal effects
- **Best for**: Understanding why annotations are needed

## Data Augmentation

The system includes sophisticated data augmentation that:

- **Preserves semantics**: Keeps original code logic intact
- **Adds variety**: Introduces syntactically correct but irrelevant code
- **Increases robustness**: Helps models generalize to diverse code patterns
- **Maintains structure**: Preserves CFG properties for graph-based models

Augmentation methods include:
- Random method insertion
- Variable declaration injection
- Control flow statement addition
- Expression complexity variation

## File Structure

```
CFWR/
├── src/main/java/cfwr/          # Java resolver and slicer integration
├── specimin/                    # Specimin slicer submodule
├── models/                      # Trained model files
├── slices/                      # Original code slices
├── slices_aug/                  # Augmented slices (auto-generated)
├── cfg_output/                  # Generated CFGs
├── hgt.py                       # HGT training script
├── gbt.py                       # GBT training script
├── causal_model.py              # Causal model training script
├── predict_hgt.py              # HGT prediction script
├── predict_gbt.py              # GBT prediction script
├── predict_causal.py           # Causal model prediction script
├── predict_on_project.py       # Comprehensive project prediction
├── augment_slices.py           # Data augmentation script
├── pipeline.py                 # End-to-end pipeline script
└── requirements.txt             # Python dependencies
```

## Examples

### Complete Workflow Example

```bash
# 1. Generate slices from Checker Framework warnings
./gradlew runResolver -Pargs="/home/user/project warnings.out /home/user/CFWR specimin"

# 2. Augment the slices for better training
python3 augment_slices.py --slices_dir slices --output_dir slices_aug --variants 3

# 3. Train all models (automatically uses augmented slices)
python3 hgt.py
python3 gbt.py  
python3 causal_model.py

# 4. Test predictions on a target project
python3 predict_on_project.py --project_root /home/user/target-project \
                              --models_dir models \
                              --output_dir predictions \
                              --slicer specimin \
                              --models hgt gbt causal
```

### Quick Test Example

```bash
# Test individual prediction scripts
python3 test_predictions.py  # Automated testing on sample files
```

## Troubleshooting

### Common Issues

1. **"No slices directory found"**: Ensure slices are generated and SLICES_DIR is set correctly
2. **"Model not found"**: Train models first using the training scripts
3. **"ClassNotFoundException"**: Ensure Checker Framework JARs are in the classpath
4. **Parsing warnings**: Some augmented slices may have parsing issues - this is normal and handled gracefully

### Debug Mode

Enable verbose output for debugging:

```bash
export CFWR_DEBUG=1
python3 hgt.py  # Will show detailed progress information
```
