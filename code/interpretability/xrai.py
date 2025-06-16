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

task_to_idx = {
        "nature_visual": 0,
        "nep_materiality_visual": 1,
        "nep_biological_visual": 2,
        "landscape-type_visual": 3
    }

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


def apply_xrai(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    num_images: int = 10,
    task: str = "nature_visual",
    steps: int = 50,
    alpha: float = 0.5
):
    
    idx = task_to_idx.get(task)
    if idx is None:
        raise ValueError(f"Unknown task '{task}'")

    wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
    explainer = XRAI(wrapper, steps=steps)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        preds = wrapper(imgs).argmax(1).cpu().tolist()
        for img_tensor, name, cls in zip(imgs, names, preds):
            if saved >= num_images:
                return
            heatmap = explainer.explain(img_tensor.unsqueeze(0), cls)

            orig_bgr = np.uint8(255 * denormalize_image(img_tensor.cpu()))[..., ::-1]
            mask = cv2.resize((heatmap*255).astype(np.uint8),
                              (orig_bgr.shape[1], orig_bgr.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            heatc = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatc, alpha, orig_bgr, 1-alpha, 0)

            cv2.imwrite(
                os.path.join(output_dir, f"{name}_{task}_xrai.png"),
                overlay
            )
            saved += 1


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
    if tasks is None:
        tasks = list(task_to_idx.keys())
    for task in tasks:
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        apply_xrai(
            multi_model, device, data_loader, task_dir,
            num_images, task, steps, alpha
        )




class XRAISmoothed:
    def __init__(
        self,
        model: torch.nn.Module,
        steps: int = 50,
        nt_samples: int = 30,
        nt_stdev: float = 0.1,
        segmentation_func: Optional[Callable] = None
    ):
        self.model = model.eval()
        self.ig = IntegratedGradients(self.model)
        self.nt = NoiseTunnel(self.ig)
        self.steps = int(steps)
        # Ensure at least 1 sample
        self.nt_samples = max(1, int(nt_samples))
        self.nt_stdev = float(nt_stdev)
        self.seg_fn = segmentation_func or (lambda img: felzenszwalb(img, scale=80, sigma=0.5, min_size=50))

    def explain(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        baseline = torch.zeros_like(input_tensor)
        atts = self.nt.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=self.steps,
            nt_type='smoothgrad',
            stdevs=self.nt_stdev,
            nt_samples=self.nt_samples,
            nt_samples_batch_size=1
        )
        attr = atts.squeeze(0).sum(0).cpu().clamp(min=0).numpy()
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        img_uint = (denormalize_image(input_tensor[0].cpu()) * 255).astype(np.uint8)
        segments = self.seg_fn(img_uint)
        hm = np.zeros_like(attr)
        for seg_val in np.unique(segments):
            mask = segments == seg_val
            hm[mask] = attr[mask].mean()
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        return hm


def apply_xrai_smoothed(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    num_images: int = 10,
    task: str = "nature_visual",
    steps: int = 50,
    nt_samples: int = 30,
    nt_stdev: float = 0.1,
    alpha: float = 0.6,
    morph_kernel: int = 5
):
    os.makedirs(output_dir, exist_ok=True)
    idx = task_to_idx.get(task)
    if idx is None:
        raise ValueError(f"Unknown task {task}")
    wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
    explainer = XRAISmoothed(wrapper, steps, nt_samples, nt_stdev)
    saved = 0
    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        preds = wrapper(imgs).argmax(1).cpu().tolist()
        for img_tensor, name, cls in zip(imgs, names, preds):
            if saved >= num_images:
                return
            hm = explainer.explain(img_tensor.unsqueeze(0), cls)
            # Morphology filtering
            hm_u8 = (hm * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            hm_u8 = cv2.morphologyEx(hm_u8, cv2.MORPH_OPEN, kernel)
            # Overlay
            orig_bgr = np.uint8(255 * denormalize_image(img_tensor.cpu()))[..., ::-1]
            mask = cv2.resize(hm_u8, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
            cmap = cv2.COLORMAP_PLASMA
            heatc = cv2.applyColorMap(mask, cmap)
            overlay = cv2.addWeighted(heatc, alpha, orig_bgr, 1 - alpha, 0)
            # Contours
            thresh = np.percentile(mask, 90)
            bin_m = (mask >= thresh).astype(np.uint8)
            contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(output_dir, f"{name}_{task}_xrai.png"), overlay)
            saved += 1


def apply_xrai_smoothed_all(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    tasks: list = None,
    num_images: int = 10,
    steps: int = 50,
    alpha: float = 0.5
):
    if tasks is None:
        tasks = list(task_to_idx.keys())
    for task in tasks:
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        apply_xrai_smoothed(
            multi_model, device, data_loader, task_dir,
            num_images, task, steps, alpha
        )

