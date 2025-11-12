import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    def __init__(self, path, stats_path, custom_size=10_000):
        self.custom_size = custom_size
        self.data = np.load(path, mmap_mode='r')
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.mean = stats["mean"]
        self.std = stats["std"]
        self.n_samples = self.data.shape[0]

    def __len__(self):
        return min(self.custom_size, self.n_samples)

    def __getitem__(self, idx):
        arr = self.data[idx].astype(np.float32)
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr).unsqueeze(0)
