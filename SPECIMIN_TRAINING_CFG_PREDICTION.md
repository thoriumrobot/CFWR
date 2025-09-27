# Specimin Training + CFG Builder Prediction Setup

This guide explains how to use **Specimin for training** and **Checker Framework CFG Builder for prediction** as requested.

## Overview

The CFWR system has been configured to support different slicers for different phases:
- **Training Phase**: Uses Specimin slicer for generating training data
- **Prediction Phase**: Uses Checker Framework CFG Builder for generating prediction data

## Quick Start

### Complete Workflow (Recommended)

Run the complete workflow with a single command:

```bash
python specimin_train_cfg_predict.py \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/CFWR/index1.out \
  --models hgt gbt causal \
  --output_dir /home/ubuntu/CFWR/final_results
```

This will:
1. Train models using Specimin slicer
2. Run predictions using CFG Builder slicer
3. Place annotations using the predictions

### Step-by-Step Approach

#### 1. Training with Specimin

```bash
python train_with_specimin.py \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/CFWR/index1.out \
  --models hgt gbt causal
```

#### 2. Prediction with CFG Builder

```bash
python predict_with_cfg_builder.py \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/CFWR/index1.out \
  --models hgt gbt causal \
  --output_dir /home/ubuntu/CFWR/prediction_results_cf
```

## Directory Structure

The system creates separate directories for each slicer:

```
CFWR/
├── slices_specimin/           # Specimin-generated slices (training)
├── slices_aug_specimin/       # Augmented Specimin slices (training)
├── cfg_output_specimin/       # CFGs from Specimin slices (training)
├── models_specimin/           # Models trained with Specimin data
├── slices_cf/                 # CFG Builder-generated slices (prediction)
├── slices_aug_cf/             # Augmented CFG Builder slices (prediction)
├── cfg_output_cf/             # CFGs from CFG Builder slices (prediction)
└── predictions_cf/            # Predictions using CFG Builder data
```

## Environment Variables

The scripts automatically set these environment variables:

### Training (Specimin)
- `SLICES_DIR=/home/ubuntu/CFWR/slices_specimin`
- `AUGMENTED_SLICES_DIR=/home/ubuntu/CFWR/slices_aug_specimin`
- `CFG_OUTPUT_DIR=/home/ubuntu/CFWR/cfg_output_specimin`
- `MODELS_DIR=/home/ubuntu/CFWR/models_specimin`

### Prediction (CFG Builder)
- `SLICES_DIR=/home/ubuntu/CFWR/slices_cf`
- `AUGMENTED_SLICES_DIR=/home/ubuntu/CFWR/slices_aug_cf`
- `CFG_OUTPUT_DIR=/home/ubuntu/CFWR/cfg_output_cf`
- `MODELS_DIR=/home/ubuntu/CFWR/models_specimin` (uses Specimin-trained models)

## Key Features

### 1. Separate Slicer Configuration
- **Training**: Uses Specimin slicer (`--slicer specimin`)
- **Prediction**: Uses CFG Builder slicer (`--slicer cf`)

### 2. Model Reuse
- Models are trained with Specimin data
- Same models are used for prediction with CFG Builder data
- This tests model generalization across different slicers

### 3. Automatic Directory Management
- Creates separate directories for each slicer
- Prevents data mixing between training and prediction
- Maintains clear separation of concerns

### 4. Comprehensive Logging
- Step-by-step progress reporting
- Clear success/failure indicators
- Detailed error messages

## Advanced Usage

### Skip Training (Use Existing Models)

```bash
python specimin_train_cfg_predict.py \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/CFWR/index1.out \
  --skip_training \
  --output_dir /home/ubuntu/CFWR/final_results
```

### Skip Prediction

```bash
python specimin_train_cfg_predict.py \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/CFWR/index1.out \
  --skip_prediction \
  --output_dir /home/ubuntu/CFWR/final_results
```

### Skip Annotation Placement

```bash
python specimin_train_cfg_predict.py \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/CFWR/index1.out \
  --skip_annotation_placement \
  --output_dir /home/ubuntu/CFWR/final_results
```

## Troubleshooting

### Specimin Issues
- Ensure `SPECIMIN_JARPATH` is set correctly
- Check that Checker Framework jars are accessible
- Verify Specimin submodule is initialized

### CFG Builder Issues
- Ensure `CHECKERFRAMEWORK_CP` is set correctly
- Check that CFG Builder jar is built (`build/libs/cf-slicer-all.jar`)
- Verify Java 21+ is being used

### Model Loading Issues
- Check that models exist in `models_specimin/` directory
- Verify model file names match expected patterns
- Ensure models were trained successfully

## Expected Results

### Training Phase
- Slices generated in `slices_specimin/`
- Augmented slices in `slices_aug_specimin/`
- CFGs in `cfg_output_specimin/`
- Trained models in `models_specimin/`

### Prediction Phase
- Slices generated in `slices_cf/`
- Augmented slices in `slices_aug_cf/`
- CFGs in `cfg_output_cf/`
- Predictions in `predictions_cf/`

### Final Results
- Annotated project in specified output directory
- Annotation placement report
- Pipeline summary with statistics

## Benefits of This Approach

1. **Slicer Diversity**: Tests model robustness across different slicing approaches
2. **Quality Comparison**: Allows comparison of Specimin vs CFG Builder slice quality
3. **Flexibility**: Easy to switch between slicers for different phases
4. **Reproducibility**: Clear separation makes results reproducible
5. **Debugging**: Easier to debug issues in specific phases

This configuration fulfills your requirement to use Specimin for training and CFG Builder for prediction while maintaining the full functionality of the CFWR system.
