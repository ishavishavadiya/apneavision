import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ApneaSpectrogramDataset(Dataset):
    def __init__(self, processed_folder="data/processed", transform=None):
        self.files = [os.path.join(processed_folder, f) for f in os.listdir(processed_folder) if f.endswith(".npy")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # shape: (channels, freq, time)
        
        # Example: use only first channel for now
        spec = spec[0]  
        
        # Expand dims to fit CNN (C,H,W)
        spec = np.expand_dims(spec, axis=0).astype(np.float32)

        # Parse label from filename (your preprocess saved label in name)
        fname = os.path.basename(self.files[idx])
        label = int(fname.split("_label")[-1].split(".")[0])

        if self.transform:
            spec = self.transform(spec)

        return torch.tensor(spec), torch.tensor(label, dtype=torch.long)
