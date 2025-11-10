import torch
import os
from tqdm import tqdm

CODES_DIR = '/datasets/onailana/codes'

for f in tqdm(os.listdir(CODES_DIR)):
    if f.endswith(".pt"):
        path = os.path.join(CODES_DIR, f)
        data = torch.load(path)

        if "mask" not in data:
            degraded = data["degraded"]
            data["mask"] = torch.ones_like(degraded, dtype=torch.bool)
            torch.save(data, path)

