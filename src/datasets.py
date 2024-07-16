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
        #mean = x.mean(dim=-1, keepdim=True)
        return x - mean

    X = baseline_correction(X)
    X = scaling(X)

    return X


def add_subject_info(X, subject_id):
    idxs = torch.zeros(X.shape[0], X.shape[1], 4)
    for i in range(X.shape[0]):
        idx = subject_id[i]
        for j in range(X.shape[1]):
            idxs[i][j][idx] = 1

    return torch.cat((X, idxs), 2)


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        #print("start loading X")
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.X = preprocess(self.X)
        #print("finish loading X")

        #print("start loading subject_idxs")
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        #print("finish loading subject_idxs")

        #self.X = add_subject_info(self.X, self.subject_idxs)   #ここで被験者情報追加？
        
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