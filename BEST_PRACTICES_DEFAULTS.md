# CFWR Best Practices Defaults

This document outlines the best practices that are now the default behavior throughout the CFWR codebase.

## 🎯 **Core Best Practices**

### **1. Augmented Slices as Default**
- **Training**: All training scripts prefer augmented slices over original slices
- **Prediction**: All prediction scripts prefer augmented slices when available
- **Pipeline**: The main pipeline uses augmented slices by default
- **Benefit**: Improved model generalization through diverse training data

### **2. Dataflow-Augmented CFGs as Default**
- **CFG Generation**: Always includes dataflow edges connecting variables of the same name
- **Edge Types**: Different edge types for control flow vs dataflow
- **Training**: All models use dataflow information for better predictions
- **Prediction**: All prediction scripts expect dataflow-augmented CFGs
- **Benefit**: Richer graph representation for better model performance

### **3. Checker Framework Slicer as Default**
- **Slicing**: Uses Checker Framework slicer (`cf`) as the default option
- **Integration**: Seamlessly integrates with Checker Framework toolchain
- **Quality**: Produces higher quality slices compared to alternatives
- **Benefit**: Better slice quality leads to better model training

### **4. Pipeline Consistency**
- **Training**: Uses same pipeline functions as prediction
- **CFG Generation**: Consistent CFG format across all components
- **Data Flow**: Seamless data flow from slicing to prediction
- **Benefit**: Ensures consistency between training and inference

## 📁 **Updated Files**

### **Prediction Scripts**
- `predict_hgt.py` - HGT predictions with best practices defaults
- `predict_gbt.py` - GBT predictions with best practices defaults  
- `predict_causal.py` - Causal model predictions with best practices defaults
- `predict_on_project.py` - Project-wide predictions with best practices defaults

### **Training Scripts**
- `hgt.py` - HGT training with best practices defaults
- `gbt.py` - GBT training with best practices defaults
- `causal_model.py` - Causal model training with best practices defaults

### **Pipeline Scripts**
- `pipeline.py` - Main pipeline with best practices defaults
- `cfg.py` - CFG generation with dataflow by default
- `augment_slices.py` - Data augmentation with best practices

### **Reinforcement Learning Scripts**
- `enhanced_rl_training.py` - RL training with augmented slices default
- `rl_training.py` - Basic RL training with best practices
- `rl_pipeline.py` - RL pipeline with best practices defaults

## 🔧 **Command-Line Options**

### **Default Behavior (Best Practices)**
```bash
# Training - uses augmented slices and dataflow CFGs by default
python hgt.py
python gbt.py  
python causal_model.py

# Prediction - uses augmented slices and dataflow CFGs by default
python predict_hgt.py --model_path model.pth --out_path predictions.json --slices_dir slices/
python predict_gbt.py --model_path model.joblib --out_path predictions.json --slices_dir slices/
python predict_causal.py --model_path model.joblib --out_path predictions.json --slices_dir slices/

# Project-wide prediction - uses best practices by default
python predict_on_project.py --project_root /path/to/project --output_dir /path/to/output
```

### **Override Options**
```bash
# Use original slices instead of augmented slices
python predict_hgt.py --use_original_slices --model_path model.pth --out_path predictions.json --slices_dir slices/

# Use original slices in project prediction
python predict_on_project.py --use_original_slices --project_root /path/to/project --output_dir /path/to/output
```

## 🏗️ **Architecture Benefits**

### **1. Consistency**
- All scripts use the same defaults
- Training and prediction pipelines are aligned
- CFG format is consistent across all components

### **2. Performance**
- Dataflow information improves model accuracy
- Augmented slices improve model generalization
- Checker Framework slicer produces better quality slices

### **3. Maintainability**
- Clear documentation of best practices
- Consistent command-line interfaces
- Easy to understand and modify

### **4. Extensibility**
- Easy to add new models with same defaults
- Consistent integration points
- Clear separation of concerns

## 📊 **Data Flow**

```
Java Files → Checker Framework Slicer → Slices
    ↓
Slices → Data Augmentation → Augmented Slices
    ↓
Augmented Slices → CFG Generation (with dataflow) → Dataflow-Augmented CFGs
    ↓
Dataflow-Augmented CFGs → Model Training → Trained Models
    ↓
Trained Models + Dataflow-Augmented CFGs → Predictions
```

## 🎯 **Key Features**

### **Dataflow Information**
- Variables of the same name are connected with dataflow edges
- Different edge types for control flow vs dataflow
- Enhanced graph representation for better predictions

### **Augmented Slices**
- Truly random, syntactically correct Java code
- Dynamic generation of methods, statements, expressions
- Configurable randomness levels
- Improved model generalization

### **Pipeline Integration**
- Seamless flow from slicing to prediction
- Consistent data formats across components
- Easy to extend and modify

## 🔍 **Verification**

To verify that best practices are being used:

1. **Check CFG files** for dataflow edges:
   ```bash
   grep -r "dataflow_edges" cfg_output/
   ```

2. **Check for augmented slices**:
   ```bash
   ls -la slices_aug_*/
   ```

3. **Verify pipeline consistency**:
   ```bash
   python pipeline.py --help
   ```

## 📝 **Migration Guide**

If you have existing scripts that don't use best practices:

1. **Update command-line arguments** to use new defaults
2. **Remove explicit flags** that are now default behavior
3. **Update documentation** to reflect new defaults
4. **Test with new defaults** to ensure compatibility

## 🚀 **Future Enhancements**

The best practices framework is designed to be extensible:

- Easy to add new slicers with same defaults
- Simple to extend CFG generation with new features
- Straightforward to add new models with consistent interfaces
- Clear path for adding new augmentation strategies

---

**Note**: All scripts now follow these best practices by default. The old behavior can still be accessed through explicit command-line flags if needed for backward compatibility.
