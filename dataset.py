import json
import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
import pickle

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


class PickleDataset(Dataset):
    def __init__(self, pickle_path, index_path, input_size, custom_size=None):
        self.pickle_path = pickle_path
        self.offsets = np.load(index_path, mmap_mode='r')
        self.n_samples = len(self.offsets)
        self.transform = v2.Resize((input_size, input_size))

        if custom_size is None:
            self.custom_size = self.n_samples
        else:
            self.custom_size = min(custom_size, self.n_samples)

    def __len__(self):
        return self.custom_size

    def __getitem__(self, idx):
        with open(self.pickle_path, "rb") as f:
            f.seek(int(self.offsets[idx]))
            metric, arr = pickle.load(f)

        arr = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        while arr.shape != torch.Size([1, 512, 512]):
            arr = self[(idx+1)%self.n_samples]
        return self.transform(arr)

if __name__ == '__main__':
    dataset = PickleDataset("../normalized_sorted.pkl", "../normalized_sorted_index.npy", 128, custom_size=20)
    print(dataset[0].shape)
