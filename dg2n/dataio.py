import os
import torch
from torch.utils.data import Dataset

class GraphDirDataset(Dataset):
    def __init__(self, dir_path: str):
        super().__init__()
        self.dir_path = dir_path
        self.files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".pt")]
        self.files.sort()
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu")

def collate_graphs(batch):
    return batch
