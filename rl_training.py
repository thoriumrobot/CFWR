#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Checker Framework Annotation Prediction

This script implements a reinforcement learning approach where:
1. The model predicts annotation locations on augmented slices
2. Annotations are placed at the predicted locations
3. Checker Framework is run to evaluate the quality
4. Feedback based on warning count changes is used to train the model
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
from torch_geometric.data import DataLoader
from collections import defaultdict
import random
from pathlib import Path

# Import our modules
from hgt import HGTModel, create_heterodata, load_cfgs
from gbt import load_cfgs as load_cfgs_gbt, extract_features_from_cfg
from causal_model import load_cfgs as load_cfgs_causal, extract_features_and_labels, parse_warnings, run_index_checker, preprocess_data
from cfg import generate_control_flow_graphs, save_cfgs
from augment_slices import augment_file

class AnnotationPlacementModel(nn.Module):
    """Model for predicting annotation placement locations"""
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(AnnotationPlacementModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head for annotation placement
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

class ReinforcementLearningTrainer:
    """Reinforcement Learning trainer for annotation placement"""
    
    def __init__(self, model_type='hgt', learning_rate=0.001, device='cpu'):
        self.model_type = model_type
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize the appropriate model
        if model_type == 'hgt':
            self.model = self._init_hgt_model()
        elif model_type == 'gbt':
            self.model = self._init_gbt_model()
        elif model_type == 'causal':
            self.model = self._init_causal_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'warning_changes': [],
            'accuracy': []
        }
        
    def _init_hgt_model(self):
        """Initialize HGT-based annotation placement model"""
        return AnnotationPlacementModel(input_dim=64, hidden_dim=128)
    
    def _init_gbt_model(self):
        """Initialize GBT-based annotation placement model"""
        return AnnotationPlacementModel(input_dim=14, hidden_dim=128)  # Based on GBT features
    
    def _init_causal_model(self):
        """Initialize Causal-based annotation placement model"""
        return AnnotationPlacementModel(input_dim=12, hidden_dim=128)  # Based on Causal features
    
    def predict_annotation_locations(self, cfg_data, threshold=0.5):
        """Predict where annotations should be placed based on CFG data"""
        if self.model_type == 'hgt':
            return self._predict_hgt_locations(cfg_data, threshold)
        elif self.model_type == 'gbt':
            return self._predict_gbt_locations(cfg_data, threshold)
        elif self.model_type == 'causal':
            return self._predict_causal_locations(cfg_data, threshold)
    
    def _predict_hgt_locations(self, cfg_data, threshold):
        """Predict annotation locations using HGT model"""
        try:
            # Create heterodata from CFG
            heterodata = create_heterodata(cfg_data)
            if heterodata is None:
                return []
            
            # Get node features
            node_features = heterodata.x
            if node_features is None or len(node_features) == 0:
                return []
            
            # Predict for each node
            self.model.eval()
            with torch.no_grad():
                logits = self.model(node_features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities[:, 1] > threshold
            
            # Extract line numbers for predicted locations
            predicted_lines = []
            for i, (pred, node) in enumerate(zip(predictions, heterodata.node_info)):
                if pred and node.get('line') is not None:
                    predicted_lines.append(node['line'])
            
            return predicted_lines
        except Exception as e:
            print(f"Error in HGT prediction: {e}")
            return []
    
    def _predict_gbt_locations(self, cfg_data, threshold):
        """Predict annotation locations using GBT model"""
        try:
            # Extract features
            features = extract_features_from_cfg(cfg_data)
            if not features:
                return []
            
            # Convert to tensor
            feature_tensor = torch.tensor([features], dtype=torch.float32)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                logits = self.model(feature_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = probabilities[0, 1] > threshold
            
            # For GBT, we predict at method level, so return method start line
            if prediction and cfg_data.get('method_name'):
                # Find method start line from CFG nodes
                for node in cfg_data.get('nodes', []):
                    if node.get('label') == 'Entry':
                        return [node.get('line', 1)]
            
            return []
        except Exception as e:
            print(f"Error in GBT prediction: {e}")
            return []
    
    def _predict_causal_locations(self, cfg_data, threshold):
        """Predict annotation locations using Causal model"""
        try:
            # Extract features
            records = extract_features_and_labels(cfg_data, {})
            if not records:
                return []
            
            # Use the first record for prediction
            features = records[0]
            feature_values = [
                features.get('label_length', 0),
                features.get('in_degree', 0),
                features.get('out_degree', 0),
                features.get('label_encoded', 0),
                features.get('line_number', 0),
                features.get('control_in_degree', 0),
                features.get('control_out_degree', 0),
                features.get('dataflow_in_degree', 0),
                features.get('dataflow_out_degree', 0),
                features.get('variables_used', 0),
                features.get('dataflow_count', 0),
                features.get('control_count', 0)
            ]
            
            # Convert to tensor
            feature_tensor = torch.tensor([feature_values], dtype=torch.float32)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                logits = self.model(feature_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = probabilities[0, 1] > threshold
            
            # Return the line number if prediction is positive
            if prediction and features.get('line_number'):
                return [features['line_number']]
            
            return []
        except Exception as e:
            print(f"Error in Causal prediction: {e}")
            return []
    
    def place_annotations(self, java_file, predicted_lines, annotation_type='@NonNull'):
        """Place annotations at predicted locations in the Java file"""
        try:
            with open(java_file, 'r') as f:
                lines = f.readlines()
            
            # Sort predicted lines in descending order to avoid line number shifts
            predicted_lines = sorted(predicted_lines, reverse=True)
            
            for line_num in predicted_lines:
                if 0 < line_num <= len(lines):
                    # Find the appropriate place to insert annotation
                    # Look for variable declarations, method parameters, or return types
                    line = lines[line_num - 1].strip()
                    
                    # Skip if line is empty or already has annotations
                    if not line or line.startswith('@'):
                        continue
                    
                    # Insert annotation before the line
                    annotation_line = f"    {annotation_type}\n"
                    lines.insert(line_num - 1, annotation_line)
            
            # Write back to file
            with open(java_file, 'w') as f:
                f.writelines(lines)
            
            return True
        except Exception as e:
            print(f"Error placing annotations: {e}")
            return False
    
    def evaluate_with_checker_framework(self, java_file):
        """Run Checker Framework on the annotated file and return warning count"""
        try:
            # Set up environment
            env = os.environ.copy()
            checker_cp = env.get('CHECKERFRAMEWORK_CP', '')
            
            # Run Checker Framework
            cmd = [
                'javac',
                '-cp', checker_cp,
                '-processor', 'org.checkerframework.checker.nullness.NullnessChecker',
                '-Xmaxwarns', '1000',
                java_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(java_file))
            
            # Parse warnings from stderr
            warning_count = 0
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if 'warning:' in line.lower():
                        warning_count += 1
            
            return warning_count, result.stderr
        except Exception as e:
            print(f"Error running Checker Framework: {e}")
            return 0, str(e)
    
    def compute_reward(self, original_warnings, new_warnings):
        """Compute reward based on warning count change"""
        if original_warnings == 0:
            # If no original warnings, reward for not introducing new ones
            return 1.0 if new_warnings == 0 else -0.5
        else:
            # Reward for reducing warnings
            reduction = original_warnings - new_warnings
            return reduction / original_warnings
    
    def train_episode(self, cfg_data, original_warnings, java_file):
        """Train the model on a single episode"""
        try:
            # Predict annotation locations
            predicted_lines = self.predict_annotation_locations(cfg_data)
            
            if not predicted_lines:
                return 0.0  # No reward if no predictions
            
            # Create a copy of the file for annotation
            temp_file = java_file + '.annotated'
            shutil.copy2(java_file, temp_file)
            
            # Place annotations
            if not self.place_annotations(temp_file, predicted_lines):
                return 0.0
            
            # Evaluate with Checker Framework
            new_warnings, stderr = self.evaluate_with_checker_framework(temp_file)
            
            # Compute reward
            reward = self.compute_reward(original_warnings, new_warnings)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return reward
            
        except Exception as e:
            print(f"Error in training episode: {e}")
            return 0.0
    
    def train(self, slices_dir, cfg_dir, num_episodes=100, batch_size=32, use_augmented_slices=True):
        """Main training loop using augmented slices (default behavior)
        
        Args:
            slices_dir: Directory containing augmented slices (default behavior)
            cfg_dir: Directory containing CFGs generated from augmented slices
            num_episodes: Number of training episodes
            batch_size: Batch size for training
            use_augmented_slices: Whether to use augmented slices (default: True)
        """
        slice_type = "augmented" if use_augmented_slices else "original"
        print(f"Starting RL training with {self.model_type} model for {num_episodes} episodes")
        print(f"Training on {slice_type} slices from: {slices_dir}")
        
        # Find all CFG files
        cfg_files = []
        for root, dirs, files in os.walk(cfg_dir):
            for file in files:
                if file.endswith('.json'):
                    cfg_files.append(os.path.join(root, file))
        
        if not cfg_files:
            print("No CFG files found for training")
            return
        
        print(f"Found {len(cfg_files)} CFG files for training")
        
        # Training loop
        for episode in range(num_episodes):
            episode_rewards = []
            episode_warning_changes = []
            
            # Sample a batch of CFG files
            batch_files = random.sample(cfg_files, min(batch_size, len(cfg_files)))
            
            for cfg_file in batch_files:
                try:
                    # Load CFG data
                    with open(cfg_file, 'r') as f:
                        cfg_data = json.load(f)
                    
                    # Find corresponding Java file
                    java_file = self._find_corresponding_java_file(cfg_file, slices_dir)
                    if not java_file or not os.path.exists(java_file):
                        continue
                    
                    # Get original warning count
                    original_warnings, _ = self.evaluate_with_checker_framework(java_file)
                    
                    # Train on this episode
                    reward = self.train_episode(cfg_data, original_warnings, java_file)
                    
                    episode_rewards.append(reward)
                    episode_warning_changes.append(original_warnings)
                    
                except Exception as e:
                    print(f"Error processing {cfg_file}: {e}")
                    continue
            
            # Update model based on episode rewards
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                self._update_model(episode_rewards)
                
                # Record statistics
                self.training_stats['episodes'].append(episode)
                self.training_stats['rewards'].append(avg_reward)
                self.training_stats['warning_changes'].append(np.mean(episode_warning_changes))
                
                print(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, "
                      f"Avg Original Warnings = {np.mean(episode_warning_changes):.1f}")
            
            # Save model checkpoint every 10 episodes
            if episode % 10 == 0:
                self.save_model(f"models/rl_{self.model_type}_episode_{episode}.pth")
        
        print("Training completed!")
        self.save_model(f"models/rl_{self.model_type}_final.pth")
        self.save_training_stats(f"models/rl_{self.model_type}_stats.json")
    
    def _find_corresponding_java_file(self, cfg_file, slices_dir):
        """Find the Java file corresponding to a CFG file"""
        try:
            # Extract method name from CFG file path
            cfg_path = Path(cfg_file)
            method_name = cfg_path.stem
            
            # Look for corresponding Java file in slices directory
            for root, dirs, files in os.walk(slices_dir):
                for file in files:
                    if file.endswith('.java'):
                        java_file = os.path.join(root, file)
                        # Check if this Java file contains the method
                        with open(java_file, 'r') as f:
                            content = f.read()
                            if method_name in content:
                                return java_file
            return None
        except Exception as e:
            print(f"Error finding Java file: {e}")
            return None
    
    def _update_model(self, rewards):
        """Update model parameters based on rewards"""
        try:
            # Convert rewards to loss (negative rewards)
            loss = -torch.tensor(rewards, dtype=torch.float32).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        except Exception as e:
            print(f"Error updating model: {e}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'learning_rate': self.learning_rate
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
    
    def save_training_stats(self, filepath):
        """Save training statistics"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"Training stats saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning Training for Annotation Placement')
    parser.add_argument('--slices_dir', required=True, help='Directory containing augmented slices')
    parser.add_argument('--cfg_dir', required=True, help='Directory containing CFGs')
    parser.add_argument('--model_type', choices=['hgt', 'gbt', 'causal'], default='hgt',
                       help='Type of model to use for RL training')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--load_model', help='Path to load existing model')
    parser.add_argument('--use_augmented_slices', action='store_true', default=True,
                       help='Use augmented slices for training (default: True)')
    parser.add_argument('--use_original_slices', action='store_true', default=False,
                       help='Use original slices instead of augmented slices')
    
    args = parser.parse_args()
    
    # Determine whether to use augmented slices
    use_augmented_slices = args.use_augmented_slices and not args.use_original_slices
    
    # Initialize trainer
    trainer = ReinforcementLearningTrainer(
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Load existing model if specified
    if args.load_model:
        trainer.load_model(args.load_model)
    
    # Start training on augmented slices (default behavior)
    trainer.train(
        slices_dir=args.slices_dir,
        cfg_dir=args.cfg_dir,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        use_augmented_slices=use_augmented_slices
    )

if __name__ == '__main__':
    main()
