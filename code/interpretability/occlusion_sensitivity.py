
# -----------------------
# OcclusionSensitivity.py
# -----------------------

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Optional
from models.wrappers import SingleOutputWrapper
from utils.denormalize_image import denormalize_image
import torchvision.transforms as transforms


task_to_idx = {
        "nature_visual": 0,
        "nep_materiality_visual": 1,
        "nep_biological_visual": 2,
        "landscape-type_visual": 3
    }

class OcclusionSensitivity:
    """
    Implements Occlusion Sensitivity by sliding a patch over the image
    and measuring the change in output score for the predicted class.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int = 32,
        stride: int = 16,
        baseline: float = 0.0,
        preprocess: callable = None
    ):
        """
        model: single-output wrapper
        patch_size: square occlusion patch size
        stride: step size
        baseline: fill value [0..1]
        preprocess: function to apply original normalization (tensor <- numpy)
        """
        self.model = model.eval()
        self.patch_size = patch_size
        self.stride = stride
        self.baseline = baseline
        # Use provided or default denormalization->normalization
        self.preprocess = preprocess or (lambda arr: torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(model.device))

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        input_tensor: 1xC,H,W normalized tensor
        target_class: class index
        returns: HxW importance map in [0,1]
        """
        # Original score
        with torch.no_grad():
            orig_logit = self.model(input_tensor)[0, target_class].item()

        # Denormalize to numpy image [0,1]
        img_np = denormalize_image(input_tensor[0].cpu())
        H, W, _ = img_np.shape

        heatmap = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.int32)

        # Slide patch
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                occluded = img_np.copy()
                occluded[y:y+self.patch_size, x:x+self.patch_size, :] = self.baseline

                # Re-normalize
                inp = self.preprocess(occluded).to(input_tensor.device)
                with torch.no_grad():
                    logit = self.model(inp)[0, target_class].item()

                diff = orig_logit - logit
                heatmap[y:y+self.patch_size, x:x+self.patch_size] += diff
                counts[y:y+self.patch_size, x:x+self.patch_size] += 1

        # Handle borders
        # slide last positions
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                y2 = min(H, y + self.patch_size)
                x2 = min(W, x + self.patch_size)
                if counts[y:y2, x:x2].sum() > 0:
                    continue  # already covered
                occluded = img_np.copy()
                occluded[y:y2, x:x2, :] = self.baseline
                inp = self.preprocess(occluded).to(input_tensor.device)
                with torch.no_grad():
                    logit = self.model(inp)[0, target_class].item()
                diff = orig_logit - logit
                heatmap[y:y2, x:x2] += diff
                counts[y:y2, x:x2] += 1

        counts = np.maximum(counts, 1)
        heatmap /= counts
        # clamp negatives and normalize
        heatmap = np.clip(heatmap, 0, None)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


def apply_occlusion(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    num_images: int = 30,
    task: str = "nature_visual",
    patch_size: int = 32,
    stride: int = 16,
    baseline: float = 0.0,
    alpha: float = 0.5
):
    """
    Generate and save occlusion sensitivity overlays.
    """
    task_to_idx = {
        "nature_visual": 0,
        "nep_materiality_visual": 1,
        "nep_biological_visual": 2,
        "landscape-type_visual": 3
    }
    idx = task_to_idx.get(task)
    if idx is None:
        raise ValueError(f"Unknown task '{task}'")

    wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
    explainer = OcclusionSensitivity(
        wrapper,
        patch_size=patch_size,
        stride=stride,
        baseline=baseline,
        preprocess=lambda img: transforms.Normalize(
            mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        )(torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device))
    )

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        preds = wrapper(imgs).argmax(1).cpu().tolist()
        for img_tensor, name, cls in zip(imgs, names, preds):
            if saved >= num_images:
                return
            heatmap = explainer.generate(img_tensor.unsqueeze(0), cls)

            # Resize via cv2
            orig_bgr = np.uint8(255 * denormalize_image(img_tensor.cpu()))[..., ::-1]
            mask = cv2.resize((heatmap*255).astype(np.uint8),
                              (orig_bgr.shape[1], orig_bgr.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            heatc = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatc, alpha, orig_bgr, 1-alpha, 0)

            cv2.imwrite(
                os.path.join(output_dir, f"{name}_{task}_occlusion.png"),
                overlay
            )
            saved += 1


def apply_occlusion_all(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    tasks: list = None,
    num_images: int = 30,
    patch_size: int = 32,
    stride: int = 16,
    baseline: float = 0.0,
    alpha: float = 0.5
):
    if tasks is None:
        tasks = list(task_to_idx.keys())
    for task in tasks:
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        apply_occlusion(
            multi_model, device, data_loader, task_dir,
            num_images, task, patch_size, stride, baseline, alpha
        )