from torch.utils.data import Dataset
import torch
import pandas as pd

class LCMDistillDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = torch.load(row["filepath"])
        
        degraded = item["degraded"]
        teacher = item["teacher"]
        mask = item["mask"]

        # metadata
        original_path = row["original_wav"]
        degraded_path = row["degraded_wav"]
        code_path = row["code_path"]
        duration = row["duration"]

        return degraded, teacher, mask, original_path, degraded_path, code_path, duration

