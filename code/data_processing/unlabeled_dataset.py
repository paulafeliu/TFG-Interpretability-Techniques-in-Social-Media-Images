"""
unlabeled_dataset.py
----------
Defines the UnlabeledImageDataset class.
"""

import os
from PIL import Image
from torch.utils.data import Dataset

class UnlabeledImageDataset(Dataset):
    """
    Dataset for loading images from a directory WITHOUT requiring a CSV.
    """
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        # We list all valid image file paths only once:
        self.img_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        name = os.path.splitext(os.path.basename(path))[0]
        return img, name, {}
