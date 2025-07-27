import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class CheXpertDataset(Dataset):
    def __init__(self, root, norm=True, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm

        self.images = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.images.extend(glob.glob(os.path.join(self.root, "**", ext), recursive=True))

        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        if self.norm:
            img = (np.asarray(img).astype(np.float32) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float32) / 255.0

        return torch.from_numpy(img).unsqueeze(0).float()

    def __len__(self):
        return len(self.images)
