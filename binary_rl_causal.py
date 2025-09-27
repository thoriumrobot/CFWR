#!/usr/bin/env python3
"""
Binary Reinforcement Learning Training Script for Causal Model
Focuses only on predicting whether an annotation needs to be placed on fields, methods, or parameters.
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
from pathlib import Path
import time
import logging

# Import our modules
from causal_model import load_cfgs as load_cfgs_causal, extract_features_and_labels
from cfg import generate_control_flow_graphs, save_cfgs
from augment_slices import augment_file
from annotation_placement import AnnotationPlacementManager
from checker_framework_integration import CheckerFrameworkEvaluator, CheckerType, EvaluationResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryCausalModel(nn.Module):
    """Binary Causal model for predicting annotation placement (yes/no)"""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(BinaryCausalModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Causal feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Binary classification head (annotation needed: yes/no)
        self.classifier = nn.Linear(hidden_dim // 2, 2)  # 2 classes: no annotation, annotation needed
        
        # Causal attention mechanism
        self.causal_attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=2, batch_first=True)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Apply causal attention if input is 3D (batch, sequence, features)
        if len(features.shape) == 3:
            attended_features, _ = self.causal_attention(features, features, features)
            features = attended_features.mean(dim=1)  # Global average pooling
        
        logits = self.classifier(features)
        return logits

class BinaryCausalTrainer:
    """Binary RL trainer for Causal-based annotation placement prediction"""
    
    def __init__(self, learning_rate=0.001, device='cpu', checker_type='index'):
        self.device = device
        self.learning_rate = learning_rate
        self.checker_type = CheckerType.INDEX if checker_type == 'index' else CheckerType.NULLNESS
        
        # Initialize the Causal-based binary model
        self.model = BinaryCausalModel(input_dim=12, hidden_dim=128)
        self.model.to(device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # Experience replay buffer for better learning
        self.experience_buffer = deque(maxlen=1000)
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'warning_changes': [],
            'accuracy': [],
            'binary_predictions': []
        }
        
        # Checker Framework evaluator
        self.evaluator = CheckerFrameworkEvaluator()
        
    def extract_causal_features(self, cfg_data):
        """Extract causal features from CFG nodes for binary classification"""
        nodes = cfg_data.get('nodes', [])
        features = []
        node_info = []
        
        for node in nodes:
            # Extract causal-style features
            feature_vector = self._extract_causal_node_features(node, cfg_data)
            features.append(feature_vector)
            
            node_info.append({
                'id': node.get('id'),
                'line': node.get('line'),
                'label': node.get('label', ''),
                'node_type': node.get('node_type', '')
            })
        
        return np.array(features), node_info
    
    def _extract_causal_node_features(self, node, cfg_data):
        """Extract causal features for a single node"""
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        line = node.get('line', 0)
        
        # Basic causal features (12 dimensions to match original causal model)
        features = [
            len(label),  # label_length
            1.0 if line > 0 else 0.0,  # has_line_number
            float('method' in node_type.lower()),  # is_method
            float('field' in node_type.lower()),  # is_field
            float('parameter' in node_type.lower()),  # is_parameter
            self._is_annotation_target(node),  # is_annotation_target
            float('public' in label.lower()),  # is_public
            float('private' in label.lower()),  # is_private
            float('static' in label.lower()),  # is_static
            float('void' in label.lower()),  # is_void
            float('int' in label.lower()),  # is_int
            float('string' in label.lower()),  # is_string
        ]
        
        return features
    
    def _is_annotation_target(self, node):
        """Determine if a node is a valid annotation target"""
        label = node.get('label', '').lower()
        node_type = node.get('node_type', '').lower()
        
        # Check for method declarations
        if any(keyword in label for keyword in ['methoddeclaration', 'constructordeclaration']):
            return 1.0
        
        # Check for field declarations
        if any(keyword in label for keyword in ['fielddeclaration', 'variabledeclarator']):
            return 1.0
        
        # Check for parameter declarations
        if any(keyword in label for keyword in ['formalparameter', 'parameter']):
            return 1.0
        
        # Check node type
        if node_type in ['method', 'field', 'parameter', 'variable']:
            return 1.0
        
        return 0.0
    
    def predict_binary_annotations(self, cfg_data, threshold=0.5):
        """Predict whether annotations are needed (binary classification)"""
        features, node_info = self.extract_causal_features(cfg_data)
        
        if len(features) == 0:
            return []
        
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(features, dtype=torch.float).to(self.device)
            logits = self.model(X)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predictions for "annotation needed" class (class 1)
            annotation_probs = probabilities[:, 1]
            predictions = annotation_probs > threshold
            
            # Extract line numbers for nodes that need annotations
            predicted_lines = []
            for i, (pred, node) in enumerate(zip(predictions, node_info)):
                if pred and node.get('line') and self._is_annotation_target({'label': node['label'], 'node_type': node['node_type']}) > 0:
                    predicted_lines.append({
                        'line': node['line'],
                        'confidence': annotation_probs[i].item(),
                        'node_type': node['node_type'],
                        'label': node['label'][:50]  # Truncate for readability
                    })
        
        return predicted_lines
    
    def place_binary_annotations(self, java_file, predicted_lines):
        """Place binary annotations (generic @NonNull or @IndexFor)"""
        try:
            # Create a copy of the file for annotation
            temp_file = java_file + '.binary_annotated'
            shutil.copy2(java_file, temp_file)
            
            # Use the annotation placement manager
            manager = AnnotationPlacementManager(temp_file)
            
            # Extract just the line numbers
            line_numbers = [pred['line'] for pred in predicted_lines]
            
            # Determine annotation category based on checker type
            annotation_category = 'index' if self.checker_type == CheckerType.INDEX else 'nullness'
            
            logger.info(f"Placing binary annotations on lines: {line_numbers}")
            logger.info(f"Annotation category: {annotation_category}")
            
            # Place annotations
            success = manager.place_annotations(line_numbers, annotation_category)
            
            logger.info(f"Binary annotation placement success: {success}")
            return temp_file if success else None
            
        except Exception as e:
            logger.error(f"Error placing binary annotations: {e}")
            return None
    
    def evaluate_with_checker_framework(self, java_file):
        """Run Checker Framework evaluation"""
        try:
            result = self.evaluator.evaluate_file(java_file, self.checker_type)
            return result
        except Exception as e:
            logger.error(f"Error running Checker Framework: {e}")
            return EvaluationResult(
                original_warnings=[],
                new_warnings=[],
                warning_count_change=0,
                success=False,
                error_message=str(e)
            )
    
    def compute_reward(self, original_warnings, new_warnings):
        """Compute reward based on warning reduction"""
        if not original_warnings:
            return 0.0
        
        reduction = len(original_warnings) - len(new_warnings)
        
        # Normalize reward between -1 and 1
        if reduction > 0:
            return min(reduction / len(original_warnings), 1.0)
        elif reduction < 0:
            return max(reduction / len(original_warnings), -1.0)
        else:
            return 0.0
    
    def train_episode(self, cfg_data, original_warnings, java_file):
        """Train the model on a single episode"""
        try:
            # Predict binary annotation locations
            predicted_lines = self.predict_binary_annotations(cfg_data)
            
            if not predicted_lines:
                logger.info("No binary annotation targets predicted")
                return 0.0, []
            
            # Place binary annotations
            annotated_file = self.place_binary_annotations(java_file, predicted_lines)
            
            if not annotated_file:
                logger.warning("Failed to place binary annotations")
                return 0.0, []
            
            # Evaluate with Checker Framework
            result = self.evaluate_with_checker_framework(annotated_file)
            
            # Compute reward
            reward = self.compute_reward(original_warnings, result.new_warnings)
            
            # Store experience for replay
            experience = {
                'cfg_data': cfg_data,
                'predicted_lines': predicted_lines,
                'reward': reward,
                'original_warnings': len(original_warnings),
                'new_warnings': len(result.new_warnings)
            }
            self.experience_buffer.append(experience)
            
            # Clean up temp file
            if os.path.exists(annotated_file):
                os.remove(annotated_file)
            
            logger.info(f"Episode completed: reward={reward:.3f}, predictions={len(predicted_lines)}")
            return reward, predicted_lines
            
        except Exception as e:
            logger.error(f"Error in training episode: {e}")
            return 0.0, []
    
    def train(self, slices_dir, cfg_dir, num_episodes=50, batch_size=16):
        """Train the binary RL Causal model"""
        logger.info(f"Starting binary RL training for Causal model")
        logger.info(f"Slices directory: {slices_dir}")
        logger.info(f"CFG directory: {cfg_dir}")
        logger.info(f"Episodes: {num_episodes}")
        
        # Load CFG data
        cfg_files = []
        for root, dirs, files in os.walk(cfg_dir):
            for file in files:
                if file.endswith('.json'):
                    cfg_path = os.path.join(root, file)
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg_data = json.load(f)
                        cfg_files.append(cfg_data)
                    except Exception as e:
                        logger.warning(f"Failed to load CFG {cfg_path}: {e}")
        
        logger.info(f"Loaded {len(cfg_files)} CFG files")
        
        if not cfg_files:
            logger.error("No CFG files found for training")
            return
        
        # Find corresponding Java files
        java_files = []
        for root, dirs, files in os.walk(slices_dir):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(java_files)} Java files")
        
        if not java_files:
            logger.error("No Java files found for training")
            return
        
        # Training loop
        episode_rewards = []
        episode_predictions = []
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            # Select random CFG and corresponding Java file
            cfg_data = random.choice(cfg_files)
            java_file = random.choice(java_files)
            
            # Get original warnings (simulate - in real scenario, this would come from Checker Framework)
            original_warnings = [f"warning_{i}" for i in range(random.randint(5, 20))]
            
            # Train episode
            reward, predictions = self.train_episode(cfg_data, original_warnings, java_file)
            
            episode_rewards.append(reward)
            episode_predictions.append(len(predictions))
            
            # Update training statistics
            self.training_stats['episodes'].append(episode + 1)
            self.training_stats['rewards'].append(reward)
            self.training_stats['binary_predictions'].append(len(predictions))
            
            # Experience replay training (every 10 episodes)
            if len(self.experience_buffer) >= batch_size and (episode + 1) % 10 == 0:
                self._train_from_experience(batch_size)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_predictions = np.mean(episode_predictions[-10:])
                logger.info(f"Episode {episode + 1}: avg_reward={avg_reward:.3f}, avg_predictions={avg_predictions:.1f}")
        
        # Save model and training statistics
        self.save_model('models/binary_rl_causal_model.pth')
        self.save_training_stats('models/binary_rl_causal_stats.json')
        
        logger.info("Binary RL training completed")
        return self.training_stats
    
    def _train_from_experience(self, batch_size):
        """Train model using experience replay"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch from experience buffer
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        # Prepare training data
        all_features = []
        all_labels = []
        
        for experience in batch:
            cfg_data = experience['cfg_data']
            reward = experience['reward']
            
            features, _ = self.extract_causal_features(cfg_data)
            
            if len(features) == 0:
                continue
            
            # Create binary labels based on reward
            # Positive reward -> annotation needed (label 1)
            # Negative/zero reward -> no annotation (label 0)
            labels = np.full(len(features), 1 if reward > 0 else 0)
            
            all_features.append(features)
            all_labels.append(labels)
        
        if not all_features:
            return
        
        # Combine all features and labels
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # Training step
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(X_tensor)
        loss = self.criterion(logits, y_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        logger.info(f"Experience replay training: loss={loss.item():.4f}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def save_training_stats(self, filepath):
        """Save training statistics"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info(f"Training stats saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Binary RL Training for Causal Model')
    parser.add_argument('--slices_dir', default='slices_aug', help='Directory containing augmented slices')
    parser.add_argument('--cfg_dir', default='cfg_output', help='Directory containing CFGs')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checker_type', default='index', choices=['index', 'nullness'], help='Checker type')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BinaryCausalTrainer(
        learning_rate=args.learning_rate,
        device=args.device,
        checker_type=args.checker_type
    )
    
    # Train the model
    stats = trainer.train(
        slices_dir=args.slices_dir,
        cfg_dir=args.cfg_dir,
        num_episodes=args.episodes
    )
    
    logger.info("Binary RL training completed successfully")

if __name__ == '__main__':
    main()
