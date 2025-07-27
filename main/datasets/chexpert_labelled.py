import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CheXpertLabelledDataset(Dataset):
    """CheXpert dataset with multi-label annotations."""

    def __init__(self, csv_path, root, transform=None, norm=True, subsample_size=None):
        if not os.path.isfile(csv_path):
            raise ValueError(f"CSV file {csv_path} does not exist")
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")

        self.root = root
        self.transform = transform
        self.norm = norm

        df = pd.read_csv(csv_path)
        # Assume label columns start after the metadata columns
        label_cols = df.columns[5:]
        self.paths = df["Path"].tolist()
        labels = df[label_cols].fillna(0).replace(-1, 0).astype(np.float32)
        self.labels = labels.values

        if subsample_size is not None:
            self.paths = self.paths[:subsample_size]
            self.labels = self.labels[:subsample_size]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.paths[idx])
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        img = np.asarray(img).astype(np.float32)
        if self.norm:
            img = img / 127.5 - 1.0
        else:
            img = img / 255.0

        label = torch.from_numpy(self.labels[idx])
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img, label

    def __len__(self):
        return len(self.paths)
