from torch.utils.data import Dataset
import torch
import pandas as pd

class LCMDistillDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = torch.load(self.df.iloc[idx]["filepath"])
        return item["degraded"], item["teacher"]  # (input, target)

dataset = LCMDistillDataset('code_data.csv')
