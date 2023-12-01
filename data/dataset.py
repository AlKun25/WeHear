from typing import Any
import torch 
from torch.utils.data import ConcatDataset, Dataset
import torchvision.transforms as transforms 
import numpy as np
import os

def sliding_window(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(0, self.frame_length, self.hop_length)

class EEGDataset(Dataset):
    def __init__(self, files: list, frame_length:int, hop_length:int) -> None:
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.files = files
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index) -> Any:
        # return super().__getitem__(index)
        pass
    
if __name__ == "__main__":
    t = torch.from_numpy(np.load("/home/kunal/eeg_data/derivatives/toy_split/test/test_-_sub-001_-_audiobook_3_-_mel.npy"))
    window_len = 64
    hop_len = 64
    t_dash = t.unfold(0,64,64)
    
    print(t_dash)
    print(t_dash.shape)