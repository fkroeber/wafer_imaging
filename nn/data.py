import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# define csv dataset
class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, label_map=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map or {
            label: idx for idx, label in enumerate(sorted(df["label"].unique()))
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        img = torch.tensor(np.array(image)).permute(2, 0, 1) / 255
        label = self.label_map[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


# custom rotation transform
class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
