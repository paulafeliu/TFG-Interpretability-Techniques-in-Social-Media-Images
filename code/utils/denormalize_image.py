import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

def denormalize_image(tensor_image):
    """
    Converts a normalized tensor image back to a NumPy array in [0,1].
    """
    image = tensor_image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    return np.clip(image, 0, 1)