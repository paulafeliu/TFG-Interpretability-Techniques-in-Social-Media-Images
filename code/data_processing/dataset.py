"""
dataset.py
----------
Defines the MultiTaskImageDataset class for loading images and labels.
"""
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MultiTaskImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, label_maps=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label_maps = label_maps

        # Read the CSV file containing image references.
        data = pd.read_csv(csv_file).fillna("nan")
        extensions = ['.jpg', '.png', '.jpeg']
        valid_rows = []

        for idx, row in data.iterrows():
            base_img_name = row['image_name']
            # Check if at least one image file with the expected extensions exists.
            if any(os.path.exists(os.path.join(self.img_dir, base_img_name + ext)) for ext in extensions):
                valid_rows.append(row)
        # Only keep rows with valid image files.
        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_img_name = row['image_name']
        extensions = ['.jpg', '.png', '.jpeg']
        img_path = None

        for ext in extensions:
            candidate = os.path.join(self.img_dir, base_img_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = {}
        for task in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']:
            label_value = row[task]
            if self.label_maps and task in self.label_maps:
                label_value = self.label_maps[task].get(label_value, -1)
            labels[task] = label_value

        return image, row['image_name'], labels


class InferenceImageDataset(Dataset):

    def __init__(self, img_dir: Path, transform, filter_idx: list[int] = None):
        self.img_files = list(img_dir.glob('*.*'))
        if filter_idx:
            self.img_files = [f for idx, f in enumerate(self.img_files) if idx in filter_idx]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        return image, img_path.stem, None
