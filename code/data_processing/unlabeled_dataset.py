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
    Dataset para cargar imágenes de un directorio SIN necesitar un CSV.
    __getitem__ devuelve: (imagen_transformada, image_name, etiquetas_vacías)
    """
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        # Listamos una sola vez todas las rutas de imagen válidas:
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
        # El “image_name” sin extensión, igual que en tu otro dataset:
        name = os.path.splitext(os.path.basename(path))[0]
        # Tercero: un dict vacío (no usamos etiquetas para interpretabilidad):
        return img, name, {}
