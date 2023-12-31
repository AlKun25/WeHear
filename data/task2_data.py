import os
import json
import torch
import lightning as L
import torch.utils.data as data
from lightning.pytorch.demos.boring_classes import RandomDataset

class RegressionDataModule(L.LightningDataModule):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        with open(config_path) as fp:
            self.config = json.load(fp)
            
        self.data_dir = os.path.join(self.config["dataset_folder"],self.config["derivatives_folder"],  self.config["split_folder"])
        self.features = ["eeg",  "mel"]
        
        self.batch_size = 64
        self.frame_length = 64
        self.hop_length = 64
        
    
    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset = RandomDataset(1, 100)
        self.train, self.val, self.test = data.random_split(
            dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val)

    def test_dataloader(self):
        return data.DataLoader(self.test)

    def teardown(self):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...