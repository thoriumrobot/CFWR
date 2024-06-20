# hgt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import random
import numpy as np

class HGTModel(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super(HGTModel, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=-1,
                out_channels=hidden_channels,
                metadata={'ast_node': ['type'], 'cfg': ['edge_index']},
                num_heads=num_heads
            )
            self.convs.append(conv)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return self.fc(x_dict['ast_node'])

class IndexCheckerEnv:
    def __init__(self, model, data_loader, index_checker):
        self.model = model
        self.data_loader = data_loader
        self.index_checker = index_checker

    def reset(self):
        self.current_data = random.choice(list(self.data_loader))
        self.state = self.current_data.x_dict, self.current_data.edge_index_dict
        self.current_data['annotations'] = []
        return self.state

    def step(self, action):
        self.place_annotation(action)

        num_warnings = self.index_checker.run(self.current_data)
        reward = -num_warnings
        done = self.check_done()
        self.state = self.current_data.x_dict, self.current_data.edge_index_dict
        return self.state, reward, done

    def place_annotation(self, action):
        annotation_location = self.current_data['ast_node'][action]
        self.current_data['annotations'].append(annotation_location)

    def check_done(self):
        return len(self.current_data['annotations']) >= len(self.current_data['ast_node'])

class Agent:
    def __init__(self, model, env, lr=0.01):
        self.model = model
        self.env = env
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def select_action(self, state):
        x_dict, edge_index_dict = state
        logits = self.model(x_dict, edge_index_dict)
        probabilities = nn.Softmax(dim=-1)(logits)
        action = torch.argmax(probabilities).item()
        return action

    def train(self, num_episodes):
        best_reward = float('-inf')
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_model(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Episode {episode}: New best model saved with reward {total_reward}")

    def update_model(self, state, action, reward, next_state):
        self.optimizer.zero_grad()
        x_dict, edge_index_dict = state
        logits = self.model(x_dict, edge_index_dict)
        loss = self.criterion(logits, torch.tensor([action]))
        loss.backward()
        self.optimizer.step()

def main(data_loader, index_checker, num_episodes=100):
    hidden_channels = 64
    out_channels = 2
    num_heads = 4
    num_layers = 3
    
    model = HGTModel(hidden_channels, out_channels, num_heads, num_layers)
    env = IndexCheckerEnv(model, data_loader, index_checker)
    agent = Agent(model, env)
    
    agent.train(num_episodes)

class MockDataLoader:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

class MockIndexChecker:
    def run(self, data):
        return len(data['annotations'])

data_loader = MockDataLoader([
    HeteroData(
        x_dict={'ast_node': torch.randn(10, 64)},
        edge_index_dict={'cfg': torch.tensor([[0, 1], [1, 2]])}
    )
])
index_checker = MockIndexChecker()

main(data_loader, index_checker)
