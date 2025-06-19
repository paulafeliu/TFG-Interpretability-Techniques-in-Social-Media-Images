# -----------------------------------
# xrai.py
# -----------------------------------

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Optional
from models.wrappers import SingleOutputWrapper
from utils.denormalize_image import denormalize_image
import torchvision.transforms as transforms
from skimage.segmentation import felzenszwalb
from captum.attr import IntegratedGradients, NoiseTunnel

class XRAI:
    """
    Implements XRAI: region-based attribution using Integrated Gradients.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        steps: int = 50,
        segmentation_func: Callable = None
    ):
        self.model = model.eval()
        self.ig = IntegratedGradients(self.model)
        self.steps = steps
        # Default to Felzenszwalb segments
        self.seg_fn = segmentation_func or (lambda img: felzenszwalb(img, scale=100, sigma=0.5, min_size=50))

    def explain(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        # Compute attributions
        baseline = torch.zeros_like(input_tensor)
        atts = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=self.steps
        )
        attr_map = atts.squeeze(0).sum(0).cpu().clamp(min=0).numpy()
        # Normalize
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        # Segment on denormalized image
        img_uint = (denormalize_image(input_tensor[0].cpu())*255).astype(np.uint8)
        segments = self.seg_fn(img_uint)
        heatmap = np.zeros_like(attr_map)

        # Aggregate per region
        for seg_val in np.unique(segments):
            mask = segments == seg_val
            heatmap[mask] = attr_map[mask].mean()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

def apply_xrai_all(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    tasks: list = None,
    num_images: int = 10,
    steps: int = 50,
    alpha: float = 0.5
):
    """
    Generate & save XRAI overlays for one or more tasks.
    """
    task_to_index = {
        "nature_visual":          0,
        "nep_materiality_visual": 1,
        "nep_biological_visual":  2,
        "landscape-type_visual":  3,
    }
    if tasks is None:
        tasks = list(task_to_index.keys())

    os.makedirs(output_dir, exist_ok=True)
    saved = {t: 0 for t in tasks}

    for task in tasks:
        idx = task_to_index.get(task)
        if idx is None:
            raise ValueError(f"Unknown task '{task}'")

        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
        explainer = XRAI(wrapper, steps=steps)

        for imgs, names, _ in data_loader:
            imgs = imgs.to(device)
            preds = wrapper(imgs).argmax(1).cpu().tolist()

            for img_tensor, name, cls in zip(imgs, names, preds):
                if saved[task] >= num_images:
                    break

                heat = explainer.explain(img_tensor.unsqueeze(0), cls)
                orig_bgr = np.uint8(255 * denormalize_image(img_tensor.cpu()))[..., ::-1]
                mask = cv2.resize(
                    (heat * 255).astype(np.uint8),
                    (orig_bgr.shape[1], orig_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                heatc   = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(heatc, alpha, orig_bgr, 1 - alpha, 0)
                outpath = os.path.join(task_dir, f"{name}_{task}_xrai.png")
                cv2.imwrite(outpath, overlay)

                saved[task] += 1
            if saved[task] >= num_images:
                break

