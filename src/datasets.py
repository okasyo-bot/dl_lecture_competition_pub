import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint


def preprocess(X):

    def scaling(x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-10
        return (x - mean) / std

    def baseline_correction(x):
        mean = x[:,:,0:100].mean(dim=-1, keepdim=True)
        return x - mean

    X = baseline_correction(X)
    X = scaling(X)

    return X


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.X = preprocess(self.X) #前処理(ベースライン補正・スケーリング)

        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]