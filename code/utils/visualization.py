"""
visualization.py
----------------
Utility functions for image visualization and saving.
"""
import matplotlib.pyplot as plt
import numpy as np


def imshow_tensor(image_tensor, title=None, save_path=None):
    """
    Displays (or saves) an image tensor after de-normalization.
    """
    image = image_tensor.cpu().numpy().transpose((1, 2, 0))
    # These normalization parameters should match the ones used in transforms.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title:
        plt.title(title, fontsize=8)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
