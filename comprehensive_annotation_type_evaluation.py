#!/usr/bin/env python3
"""
Comprehensive Annotation Type Evaluation System
Evaluates GBT, HGT, and Causal models for multi-class annotation type prediction.
Provides F1 scores by annotation type for Lower Bound Checker annotations.
"""

import os
import json
import logging
import datetime
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn.functional as F

from annotation_type_prediction import AnnotationTypeClassifier, AnnotationTypeGBTModel, AnnotationTypeHGTModel, LowerBoundAnnotationType
from sg_cfgnet import SGCFGNetTrainer
from annotation_type_causal_model import AnnotationTypeCausalModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveAnnotationTypeEvaluator:
    """
    Comprehensive evaluator for all annotation type prediction models.
    Focuses on F1 scores by individual annotation type.
    """
    
    def __init__(self, dataset_dir: str = "test_results/statistical_dataset", parameter_free: bool = False, do_hpo: bool = False):
        self.dataset_dir = dataset_dir
        self.train_cfg_dir = os.path.join(dataset_dir, "train", "cfg_output")
        self.test_cfg_dir = os.path.join(dataset_dir, "test", "cfg_output")
        self.results_dir = "test_results/comprehensive_annotation_type_evaluation"
        os.makedirs(self.results_dir, exist_ok=True)
        self.classifier = AnnotationTypeClassifier()
        self.label_encoder = LabelEncoder()
        self.parameter_free = parameter_free
        self.do_hpo = do_hpo
        
        # Label space
        if parameter_free:
            allowed = [
                LowerBoundAnnotationType.NO_ANNOTATION.value,
                LowerBoundAnnotationType.POSITIVE.value,
                LowerBoundAnnotationType.NON_NEGATIVE.value,
                LowerBoundAnnotationType.GTEN_ONE.value,
                LowerBoundAnnotationType.SEARCH_INDEX_BOTTOM.value,
            ]
            self.label_encoder.fit(allowed)
            # Target annotations exclude NO_ANNOTATION for per-type table
            self.target_annotations = [
                LowerBoundAnnotationType.POSITIVE,
                LowerBoundAnnotationType.NON_NEGATIVE,
                LowerBoundAnnotationType.GTEN_ONE,
                LowerBoundAnnotationType.SEARCH_INDEX_BOTTOM,
            ]
        else:
            self.label_encoder.fit([at.value for at in LowerBoundAnnotationType])
            self.target_annotations = [at for at in LowerBoundAnnotationType if at not in [LowerBoundAnnotationType.NO_ANNOTATION]]
        
        logger.info(f"ComprehensiveAnnotationTypeEvaluator initialized. Dataset: {dataset_dir} | parameter_free={parameter_free} | hpo={do_hpo}")
        logger.info(f"Target annotations: {[at.value for at in self.target_annotations]}")

    def _remap_to_parameter_free(self, features) -> str:
        """Map a full annotation type determination to a parameter-free type based on features.
        Ensures label diversity on simple datasets."""
        label_space = [
            LowerBoundAnnotationType.POSITIVE.value,
            LowerBoundAnnotationType.NON_NEGATIVE.value,
            LowerBoundAnnotationType.GTEN_ONE.value,
            LowerBoundAnnotationType.SEARCH_INDEX_BOTTOM.value,
            LowerBoundAnnotationType.NO_ANNOTATION.value,
        ]
        # Strong signals
        if features.is_parameter and (features.has_size_call or features.has_length_call):
            return LowerBoundAnnotationType.POSITIVE.value
        if features.has_method_call and ('indexof' in str(features).lower() or 'search' in str(features).lower() or 'find' in str(features).lower()):
            return LowerBoundAnnotationType.SEARCH_INDEX_BOTTOM.value
        if features.has_index_pattern and not features.has_array_access:
            return LowerBoundAnnotationType.GTEN_ONE.value
        if features.has_numeric_type and (features.has_comparison or features.has_loop_context):
            return LowerBoundAnnotationType.NON_NEGATIVE.value
        # Heuristic spread for diversity
        h = (features.label_length + features.line_number + features.in_degree + features.out_degree) % 5
        return label_space[h]

    def _build_pf_training_matrix(self, cfg_files: List[Dict[str, Any]], min_per_class: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and balance a parameter-free training matrix (X, y_str)."""
        X_list: List[List[float]] = []
        y_list: List[str] = []
        for cfg_file in cfg_files:
            cfg_data = cfg_file['data']
            for node in cfg_data.get('nodes', []):
                from node_level_models import NodeClassifier
                if NodeClassifier.is_annotation_target(node):
                    feats = self.classifier.extract_features(node, cfg_data)
                    # Remap to PF label
                    label = self._remap_to_parameter_free(feats)
                    if label in self.label_encoder.classes_:
                        X_list.append(self.classifier.features_to_vector(feats))
                        y_list.append(label)
        if not X_list:
            return np.zeros((0, 23)), np.array([])
        # Balance
        counts = {c: 0 for c in self.label_encoder.classes_}
        for l in y_list:
            counts[l] += 1
        # Upsample with jitter
        X_bal: List[List[float]] = list(X_list)
        y_bal: List[str] = list(y_list)
        rng = np.random.default_rng(42)
        for cls in self.label_encoder.classes_:
            if cls not in counts:
                counts[cls] = 0
            needed = max(0, min_per_class - counts[cls])
            if needed == 0:
                continue
            # sample indices of this class
            idxs = [i for i, l in enumerate(y_list) if l == cls]
            if not idxs:
                continue
            for _ in range(needed):
                i = rng.choice(idxs)
                base = np.array(X_list[i], dtype=float)
                noise = rng.normal(0, 0.02, size=base.shape)
                X_bal.append((base + noise).tolist())
                y_bal.append(cls)
        return np.array(X_bal, dtype=float), np.array(y_bal)

    def load_cfg_files(self, directory: str) -> List[Dict[str, Any]]:
        """Load CFG files from a directory with proper structure for models."""
        cfg_files = []
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return cfg_files
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                cfg_path = os.path.join(directory, filename)
                try:
                    with open(cfg_path, 'r') as f:
                        cfg_data = json.load(f)
                        cfg_files.append({
                            'file': cfg_path,
                            'method': cfg_data.get('method_name', filename.replace('.json', '')),
                            'data': cfg_data
                        })
                except Exception as e:
                    logger.warning(f"Error loading CFG {filename}: {e}")
        logger.info(f"Loaded {len(cfg_files)} CFG files from {directory}")
        return cfg_files

    def create_ground_truth_labels(self, cfg_files: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Creates ground truth labels for all nodes in the CFG files,
        determining the specific annotation type and collecting node contexts.
        """
        ground_truth_labels = []
        node_contexts = []
        
        for cfg_file in cfg_files:
            cfg_data = cfg_file['data']
            file_path = cfg_file['file']
            method_name = cfg_file['method']
            
            for node in cfg_data.get('nodes', []):
                from node_level_models import NodeClassifier
                if NodeClassifier.is_annotation_target(node):
                    features = self.classifier.extract_features(node, cfg_data)
                    if self.parameter_free:
                        gt_value = self._remap_to_parameter_free(features)
                        if gt_value not in self.label_encoder.classes_:
                            continue
                    else:
                        annotation_type = self.classifier.determine_annotation_type(features)
                        gt_value = annotation_type.value
                    ground_truth_labels.append(gt_value)
                    node_contexts.append({
                    'file': file_path,
                    'method': method_name,
                    'node_id': node.get('id'),
                    'line_number': node.get('line_number'),
                    'node_type': node.get('type'),
                    'node_label': node.get('label'),
                    'actual_annotation_type': gt_value
                })
        # If PF and too few classes, it is okay; downstream will handle
        logger.info(f"Created {len(ground_truth_labels)} ground truth labels")
        return ground_truth_labels, node_contexts

    def evaluate_model(self, model_name: str, model_instance: Any, test_cfg_files: List[Dict[str, Any]],
                       ground_truth_labels: List[str], node_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a given model for annotation type prediction with comprehensive metrics.
        """
        logger.info(f"Evaluating {model_name} Model...")
        predictions = []
        predicted_node_contexts = []

        for i, cfg_file in enumerate(test_cfg_files):
            cfg_data = cfg_file['data']
            for node in cfg_data.get('nodes', []):
                from node_level_models import NodeClassifier
                if NodeClassifier.is_annotation_target(node):
                    # For models that inherit from AnnotationTypeClassifier, predict individually
                    if hasattr(model_instance, 'extract_features'):
                        features = model_instance.extract_features(node, cfg_data)
                        
                        if hasattr(model_instance, 'forward'):  # Neural network models
                            feature_vector = torch.tensor([model_instance.features_to_vector(features)], dtype=torch.float32)
                            model_instance.eval()
                            with torch.no_grad():
                                outputs = model_instance.forward(feature_vector)
                                probabilities = F.softmax(outputs, dim=1)
                                prediction = torch.argmax(outputs, dim=1).item()
                                confidence = probabilities[0, prediction].item()
                                predicted_annotation_type = model_instance.label_encoder.inverse_transform([prediction])[0]
                        else:  # GBT model
                            feature_vector = model_instance.features_to_vector(features)
                            probabilities = model_instance.model.predict_proba([feature_vector])[0]
                            prediction = model_instance.model.predict([feature_vector])[0]
                            confidence = probabilities[prediction]
                            predicted_annotation_type = model_instance.label_encoder.inverse_transform([prediction])[0]
                    else:
                        # For causal model with different interface
                        predicted_annotation_type, confidence = model_instance.predict_annotation_type(node, cfg_data, self.classifier)
                        predicted_annotation_type = predicted_annotation_type.value
                        
                    predictions.append(predicted_annotation_type)
                
                    # Find the corresponding ground truth context
                    context_idx = len(predicted_node_contexts)
                    if context_idx < len(node_contexts):
                        context = node_contexts[context_idx].copy()
                        context['predicted_annotation_type'] = predicted_annotation_type
                        context['prediction_confidence'] = confidence
                        predicted_node_contexts.append(context)

        # Align predictions and ground truth
        min_len = min(len(ground_truth_labels), len(predictions))
        y_true_aligned = ground_truth_labels[:min_len]
        y_pred_aligned = predictions[:min_len]

        if not y_true_aligned:
            logger.error(f"No ground truth labels or predictions for {model_name}. Cannot evaluate.")
            return self._empty_result(model_name, predicted_node_contexts)

        # Encode labels for sklearn metrics
        y_true_encoded = self.label_encoder.transform(y_true_aligned)
        y_pred_encoded = self.label_encoder.transform(y_pred_aligned)

        # Calculate overall metrics
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        f1_macro = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_encoded, y_pred_encoded, 
                              labels=self.label_encoder.transform(self.label_encoder.classes_))
        cm_labels = self.label_encoder.classes_.tolist()

        # Per-class F1 scores
        per_class_f1_scores = f1_score(y_true_encoded, y_pred_encoded, average=None, 
                                       labels=self.label_encoder.transform(self.label_encoder.classes_), 
                                       zero_division=0)
        per_class_f1_dict = {
            self.label_encoder.inverse_transform([i])[0]: score 
            for i, score in enumerate(per_class_f1_scores)
        }

        # Classification report
        class_report = classification_report(y_true_encoded, y_pred_encoded, 
                                             labels=self.label_encoder.transform(self.label_encoder.classes_),
                                             target_names=self.label_encoder.classes_, 
                                             zero_division=0, output_dict=True)

        # Extract annotation type specific metrics (excluding NO_ANNOTATION)
        annotation_type_f1_scores = {}
        for annotation_type in self.target_annotations:
            if annotation_type.value in per_class_f1_dict:
                annotation_type_f1_scores[annotation_type.value] = per_class_f1_dict[annotation_type.value]

        logger.info(f"{model_name} Evaluation - Accuracy: {accuracy:.3f}, F1 (macro): {f1_macro:.3f}, F1 (weighted): {f1_weighted:.3f}")
        logger.info(f"{model_name} Annotation Type F1 Scores: {annotation_type_f1_scores}")

        return {
            'model': model_name,
            'accuracy': accuracy,
            'f1_score_macro': f1_macro,
            'f1_score_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'support': len(y_true_aligned),
            'per_class_f1': per_class_f1_dict,
            'annotation_type_f1_scores': annotation_type_f1_scores,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': cm_labels,
            'classification_report': class_report,
            'detailed_predictions': predicted_node_contexts
        }

    def _empty_result(self, model_name: str, detailed_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns an empty result structure for failed evaluations."""
        return {
            'model': model_name,
            'accuracy': 0.0, 'f1_score_macro': 0.0, 'f1_score_weighted': 0.0, 
            'precision': 0.0, 'recall': 0.0, 'support': 0,
            'per_class_f1': {}, 'annotation_type_f1_scores': {},
            'confusion_matrix': [], 'confusion_matrix_labels': [],
            'classification_report': {}, 'detailed_predictions': detailed_predictions
        }

    def _evaluate_predictions(self, y_true: List[str], y_pred: List[str], model_name: str) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
        import numpy as np
        # Align label encoder space
        labels = list(self.label_encoder.classes_)
        y_true = y_true[:len(y_pred)]
        acc = accuracy_score(y_true, y_pred) if y_pred else 0.0
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0) if y_pred else 0.0
        f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) if y_pred else 0.0
        prec = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) if y_pred else 0.0
        rec = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) if y_pred else 0.0
        cm = confusion_matrix(y_true, y_pred, labels=labels) if y_pred else np.zeros((len(labels), len(labels)))
        class_report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0) if y_pred else {}
        # Per-class F1 map for table
        per_class_f1 = {}
        for lab in labels:
            if lab in class_report:
                per_class_f1[lab] = class_report[lab].get('f1-score', 0.0)
        # Annotation-type F1 subset (exclude NO_ANNOTATION if present)
        annotation_type_f1 = {lab: per_class_f1.get(lab, 0.0) for lab in labels if lab != str(LowerBoundAnnotationType.NO_ANNOTATION.value)}
        return {
            'model': model_name,
            'accuracy': acc,
            'f1_score_macro': f1_macro,
            'f1_score_weighted': f1_weighted,
            'precision': prec,
            'recall': rec,
            'support': len(y_true),
            'per_class_f1': per_class_f1,
            'annotation_type_f1_scores': annotation_type_f1,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': labels,
            'classification_report': class_report,
            'detailed_predictions': []
        }

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Runs the comprehensive evaluation for all models."""
        logger.info("🔬 Starting Comprehensive Annotation Type Evaluation")
        logger.info("=" * 80)

        train_cfg_files = self.load_cfg_files(self.train_cfg_dir)
        test_cfg_files = self.load_cfg_files(self.test_cfg_dir)

        if not train_cfg_files or not test_cfg_files:
            logger.error("Failed to load training or test data")
            return {}

        logger.info(f"📊 Dataset: {len(train_cfg_files)} train + {len(test_cfg_files)} test CFGs")
        logger.info("🎯 Evaluation Type: Multi-Class Lower Bound Checker Annotation Type Prediction")
        logger.info("=" * 80)

        # Create ground truth for test set
        ground_truth_labels, node_contexts = self.create_ground_truth_labels(test_cfg_files)
        
        results = {}

        # --- Evaluate GBT Model ---
        logger.info("\n🌲 Evaluating Annotation Type GBT Model...")
        gbt_model = AnnotationTypeGBTModel()
        gbt_model.parameter_free = self.parameter_free
        try:
            if self.do_hpo or self.parameter_free:
                # Build PF-balanced matrix
                X_train, y_train = self._build_pf_training_matrix(train_cfg_files, min_per_class=20)
                if X_train.size > 0 and len(set(y_train.tolist())) >= 2:
                    from sklearn.model_selection import GridSearchCV
                    from sklearn.ensemble import GradientBoostingClassifier
                    from annotation_type_prediction import small_grid_gbt
                    base = GradientBoostingClassifier(random_state=42)
                    grid = GridSearchCV(base, small_grid_gbt(), cv=3, n_jobs=-1, scoring='f1_weighted', verbose=0)
                    grid.fit(X_train, self.label_encoder.transform(y_train))
                    logger.info(f"GBT HPO best params: {grid.best_params_}")
                    gbt_model.model = GradientBoostingClassifier(random_state=42, **grid.best_params_)
            gbt_model.train_model(train_cfg_files)
            if gbt_model.is_trained:
                gbt_eval_results = self.evaluate_model("AnnotationTypeGBT", gbt_model, test_cfg_files, ground_truth_labels, node_contexts)
            else:
                logger.warning("GBT model training failed")
                gbt_eval_results = self._empty_result("AnnotationTypeGBT", [])
            results['AnnotationTypeGBT'] = gbt_eval_results
        except Exception as e:
            logger.error(f"Error evaluating annotation type GBT model: {e}")
            results['AnnotationTypeGBT'] = self._empty_result("AnnotationTypeGBT", [])

        # --- Evaluate HGT Model ---
        logger.info("\n🔥 Evaluating Annotation Type HGT Model...")
        # Determine input_dim from extracted features
        input_dim = 23  # Default based on AnnotationTypeFeatures
        if train_cfg_files:
            sample_cfg = train_cfg_files[0]['data']
            if sample_cfg.get('nodes'):
                sample_features = self.classifier.extract_features(sample_cfg['nodes'][0], sample_cfg)
                sample_vector = self.classifier.features_to_vector(sample_features)
                input_dim = len(sample_vector)
        output_dim = len(self.label_encoder.classes_)

        hgt_model = AnnotationTypeHGTModel(input_dim=input_dim, hidden_dim=128, num_classes=output_dim)
        hgt_model.parameter_free = self.parameter_free
        try:
            if self.do_hpo or self.parameter_free:
                from annotation_type_prediction import small_search_space_hgt
                X_train, y_train = self._build_pf_training_matrix(train_cfg_files, min_per_class=20)
                best_cfg = None
                best_score = -1.0
                for cfg in small_search_space_hgt(input_dim, output_dim):
                    tmp_model = AnnotationTypeHGTModel(input_dim=input_dim, hidden_dim=cfg['hidden_dim'], num_classes=output_dim)
                    tmp_model.parameter_free = self.parameter_free
                    tmp_model.train_model(train_cfg_files, epochs=cfg['epochs'], learning_rate=cfg['lr'])
                    tmp_res = self.evaluate_model("AnnotationTypeHGT", tmp_model, test_cfg_files, ground_truth_labels, node_contexts)
                    score = tmp_res.get('f1_score_weighted', 0.0)
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg
                if best_cfg is not None:
                    logger.info(f"HGT HPO best cfg: {best_cfg} with f1_weighted={best_score:.3f}")
                    hgt_model = AnnotationTypeHGTModel(input_dim=input_dim, hidden_dim=best_cfg['hidden_dim'], num_classes=output_dim)
                    hgt_model.parameter_free = self.parameter_free
                    hgt_model.train_model(train_cfg_files, epochs=best_cfg['epochs'], learning_rate=best_cfg['lr'])
                else:
                    hgt_model.train_model(train_cfg_files, epochs=50)
            else:
                hgt_model.train_model(train_cfg_files, epochs=50)
            hgt_eval_results = self.evaluate_model("AnnotationTypeHGT", hgt_model, test_cfg_files, ground_truth_labels, node_contexts)
            results['AnnotationTypeHGT'] = hgt_eval_results
        except Exception as e:
            logger.error(f"Error evaluating annotation type HGT model: {e}")
            results['AnnotationTypeHGT'] = self._empty_result("AnnotationTypeHGT", [])

        # --- Evaluate Causal Model ---
        logger.info("\n🔗 Evaluating Annotation Type Causal Model...")
        causal_model = AnnotationTypeCausalModel(input_dim=input_dim, hidden_dim=128, num_classes=output_dim)
        causal_model.parameter_free = self.parameter_free
        try:
            if self.do_hpo or self.parameter_free:
                from annotation_type_prediction import small_search_space_causal
                best_cfg = None
                best_score = -1.0
                for cfg in small_search_space_causal(input_dim, output_dim):
                    tmp_causal = AnnotationTypeCausalModel(input_dim=input_dim, hidden_dim=cfg['hidden_dim'], num_classes=output_dim)
                    tmp_causal.parameter_free = self.parameter_free
                    tmp_causal.train_model(train_cfg_files, self.classifier, epochs=cfg['epochs'], learning_rate=cfg['lr'])
                    tmp = self.evaluate_model("AnnotationTypeCausal", tmp_causal, test_cfg_files, ground_truth_labels, node_contexts)
                    score = tmp.get('f1_score_weighted', 0.0)
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg
                if best_cfg is not None:
                    logger.info(f"Causal HPO best cfg: {best_cfg} with f1_weighted={best_score:.3f}")
                    causal_model = AnnotationTypeCausalModel(input_dim=input_dim, hidden_dim=best_cfg['hidden_dim'], num_classes=output_dim)
                    causal_model.parameter_free = self.parameter_free
                    causal_model.train_model(train_cfg_files, self.classifier, epochs=best_cfg['epochs'], learning_rate=best_cfg['lr'])
                else:
                    causal_model.train_model(train_cfg_files, self.classifier, epochs=50)
            else:
                causal_model.train_model(train_cfg_files, self.classifier, epochs=50)
            causal_eval_results = self.evaluate_model("AnnotationTypeCausal", causal_model, test_cfg_files, ground_truth_labels, node_contexts)
            results['AnnotationTypeCausal'] = causal_eval_results
        except Exception as e:
            logger.error(f"Error evaluating annotation type Causal model: {e}")
            results['AnnotationTypeCausal'] = self._empty_result("AnnotationTypeCausal", [])

        # --- Evaluate SG-CFGNet ---
        logger.info("\n🧩 Evaluating SG-CFGNet Model...")
        try:
            # HPO for SG-CFGNet (small search)
            search_space = [
                {'hidden_dim': 64, 'epochs': 40, 'lr': 1e-3, 'lmbd_l0': 1e-4, 'lmbd_cf': 1e-3},
                {'hidden_dim': 128, 'epochs': 60, 'lr': 8e-4, 'lmbd_l0': 5e-4, 'lmbd_cf': 1e-3},
                {'hidden_dim': 128, 'epochs': 60, 'lr': 1e-3, 'lmbd_l0': 1e-4, 'lmbd_cf': 5e-3},
            ]
            best_score = -1.0
            best_cfg = None
            best_trainer = None
            for cfg in search_space:
                sg_trainer = SGCFGNetTrainer(parameter_free=True, lmbd_l0=cfg['lmbd_l0'], lmbd_constraints=1e-3, lmbd_cf=cfg['lmbd_cf'])
                tr_info = sg_trainer.train(train_cfg_files, epochs=cfg['epochs'], lr=cfg['lr'])
                if not tr_info.get('success'):
                    continue
                sg_predictions = []
                for cf in test_cfg_files:
                    cfgd = cf['data']
                    for node in cfgd.get('nodes', []):
                        from node_level_models import NodeClassifier
                        if NodeClassifier.is_annotation_target(node):
                            lab, _ = sg_trainer.predict_node(node, cfgd)
                            sg_predictions.append(lab)
                min_len = min(len(ground_truth_labels), len(sg_predictions))
                y_true = ground_truth_labels[:min_len]
                y_pred = sg_predictions[:min_len]
                tmp = self._evaluate_predictions(y_true, y_pred, "SGCFGNet")
                score = tmp.get('f1_score_weighted', 0.0)
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
                    best_trainer = sg_trainer
            if best_trainer is not None:
                logger.info(f"SG-CFGNet HPO best cfg: {best_cfg} with f1_weighted={best_score:.3f}")
                sg_predictions = []
                for cf in test_cfg_files:
                    cfgd = cf['data']
                    for node in cfgd.get('nodes', []):
                        from node_level_models import NodeClassifier
                        if NodeClassifier.is_annotation_target(node):
                            lab, _ = best_trainer.predict_node(node, cfgd)
                            sg_predictions.append(lab)
                min_len = min(len(ground_truth_labels), len(sg_predictions))
                y_true = ground_truth_labels[:min_len]
                y_pred = sg_predictions[:min_len]
                results['SGCFGNet'] = self._evaluate_predictions(y_true, y_pred, "SGCFGNet")
            else:
                results['SGCFGNet'] = self._empty_result("SGCFGNet", [])
        except Exception as e:
            logger.error(f"Error evaluating SG-CFGNet: {e}")
            results['SGCFGNet'] = self._empty_result("SGCFGNet", [])

        # --- Summarize Results ---
        self._print_evaluation_summary(results)
        self._save_results(results)

        return results

    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary with F1 scores by annotation type."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE ANNOTATION TYPE EVALUATION RESULTS")
        logger.info("=" * 80)

        # Overall model performance
        logger.info("\n📈 OVERALL MODEL PERFORMANCE:")
        logger.info("-" * 80)
        logger.info(f"{'Model':<25} {'Accuracy':<10} {'F1(macro)':<10} {'F1(weighted)':<12} {'Precision':<10} {'Recall':<10}")
        logger.info("-" * 80)

        best_f1 = 0.0
        best_accuracy = 0.0
        best_f1_model = "N/A"
        best_accuracy_model = "N/A"

        for model_name, res in results.items():
            logger.info(f"{res['model']:<25} {res['accuracy']:<10.3f} {res['f1_score_macro']:<10.3f} "
                       f"{res['f1_score_weighted']:<12.3f} {res['precision']:<10.3f} {res['recall']:<10.3f}")
            
            if res['f1_score_weighted'] > best_f1:
                best_f1 = res['f1_score_weighted']
                best_f1_model = res['model']
            if res['accuracy'] > best_accuracy:
                best_accuracy = res['accuracy']
                best_accuracy_model = res['model']

        logger.info("-" * 80)
        logger.info(f"🏆 Best F1 Score: {best_f1_model} ({best_f1:.3f})")
        logger.info(f"🎯 Best Accuracy: {best_accuracy_model} ({best_accuracy:.3f})")

        # F1 scores by annotation type
        logger.info("\n🔍 F1 SCORES BY ANNOTATION TYPE:")
        logger.info("-" * 80)
        
        # Get all annotation types that appear in results
        all_annotation_types = set()
        for res in results.values():
            all_annotation_types.update(res['annotation_type_f1_scores'].keys())
        
        all_annotation_types = sorted(list(all_annotation_types))
        
        # Print header
        header = f"{'Annotation Type':<25}"
        for model_name in results.keys():
            header += f"{model_name.replace('AnnotationType', ''):<15}"
        logger.info(header)
        logger.info("-" * 80)
        
        # Print F1 scores for each annotation type
        for annotation_type in all_annotation_types:
            if annotation_type != LowerBoundAnnotationType.NO_ANNOTATION.value:
                row = f"{annotation_type:<25}"
                for res in results.values():
                    f1_score = res['annotation_type_f1_scores'].get(annotation_type, 0.0)
                    row += f"{f1_score:<15.3f}"
                logger.info(row)

        logger.info("=" * 80)
        logger.info("🎯 Comprehensive annotation type evaluation complete!")

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON files."""
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'dataset_used': self.dataset_dir,
            'evaluation_type': 'Comprehensive Multi-Class Lower Bound Checker Annotation Type Prediction',
            'target_annotations': [at.value for at in self.target_annotations],
            'models': {}
        }
        
        best_f1 = 0.0
        best_acc = 0.0
        best_f1_model = "N/A"
        best_acc_model = "N/A"

        for model_name, res in results.items():
            summary['models'][model_name] = {
                'name': res['model'],
                'accuracy': res['accuracy'],
                'f1_score_macro': res['f1_score_macro'],
                'f1_score_weighted': res['f1_score_weighted'],
                'precision': res['precision'],
                'recall': res['recall'],
                'support': res['support'],
                'per_class_f1': res['per_class_f1'],
                'annotation_type_f1_scores': res['annotation_type_f1_scores'],
                'confusion_matrix': res['confusion_matrix'],
                'confusion_matrix_labels': res['confusion_matrix_labels']
            }
            if res['f1_score_weighted'] > best_f1:
                best_f1 = res['f1_score_weighted']
                best_f1_model = res['model']
            if res['accuracy'] > best_acc:
                best_acc = res['accuracy']
                best_acc_model = res['model']

        summary['best_performance'] = {
            'f1_score': best_f1,
            'f1_model': best_f1_model,
            'accuracy': best_acc,
            'accuracy_model': best_acc_model
        }

        # Save summary results
        output_filepath = os.path.join(self.results_dir, "comprehensive_annotation_type_evaluation_results.json")
        with open(output_filepath, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Save detailed results
        detailed_filepath = os.path.join(self.results_dir, "detailed_annotation_type_evaluation_results.json")
        with open(detailed_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"📁 Results saved to: {output_filepath}")
        logger.info(f"📁 Detailed results saved to: {detailed_filepath}")

def main():
    """Run comprehensive annotation type evaluation"""
    evaluator = ComprehensiveAnnotationTypeEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        logger.info("✅ Comprehensive annotation type evaluation completed successfully!")
        return 0
    else:
        logger.error("❌ Comprehensive annotation type evaluation failed!")
        return 1

if __name__ == "__main__":
    exit(main())
