"""
shap.py 

"""

import os
import cv2
import numpy as np
import torch
import shap

from models.wrappers import SingleOutputWrapper
from utils.denormalize_image import denormalize_image

TASK_TO_IDX = {
    "nature_visual": 0,
    "nep_materiality_visual": 1,
    "nep_biological_visual": 2,
    "landscape-type_visual": 3
}

def apply_shap(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    num_images: int = 10,
    task: str = "nature_visual",
    background_size: int = 50,
    alpha: float = 0.5
):
    """
    Compute and save SHAP overlays for a single task.
    """
    idx = TASK_TO_IDX.get(task)
    if idx is None:
        raise ValueError(f"Unknown task '{task}'")

    # wrap model and move to device
    wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()

    # build background
    background = []
    for imgs, _, _ in data_loader:
        background.append(imgs)
        if len(background) * imgs.shape[0] >= background_size:
            break
    background = torch.cat(background, dim=0)[:background_size].to(device)

    # SHAP uses a small set of “typical” inputs as baseline
    explainer = shap.GradientExplainer(wrapper, background)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        for b, name in enumerate(names):
            if saved >= num_images:
                return

            inp = imgs[b].unsqueeze(0)  # 1×C×H×W

            # shap_values: list(len=1) of arrays (1×C×h×w)
            shap_vals = explainer.shap_values(inp)[0]  # C×h×w
            # aggregate across channels
            shap_map = np.sum(np.abs(shap_vals), axis=0)  # h×w

            # normalize to [0,1]
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)

            # overlay
            orig = denormalize_image(imgs[b].detach())  # H×W×3, float [0,1]
            H, W, _ = orig.shape
            
            # 1) resize to match the original image
            mask = cv2.resize(shap_map, (W, H))     # float

            # 2) force to 2-D single channel
            if mask.ndim == 3:
                # sometimes resize will give H×W×1; squeeze it
                mask = mask[..., 0]

            # 3) convert to uint8
            heat = np.clip(mask * 255, 0, 255).astype(np.uint8)  # now shape H×W, dtype=uint8

            # 4) confirm type & shape (DEBUGGING—feel free to remove after it works)
            assert heat.dtype == np.uint8, f"heat dtype is {heat.dtype}"
            assert heat.ndim == 2, f"heat ndim is {heat.ndim}"

            # 5) apply the JET colormap (BGR output)
            heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

            # 6) prepare the original image as uint8 BGR
            #    denormalize_image gives float RGB [0,1], so:
            orig_rgb = denormalize_image(imgs[b].detach())          # H×W×3, float
            orig_bgr = np.uint8(255 * orig_rgb)[..., ::-1]          # H×W×3, uint8 BGR

            # 7) overlay and save
            overlay = cv2.addWeighted(heatc, alpha, orig_bgr, 1 - alpha, 0)

            out_path = os.path.join(output_dir, f"{name}_{task}_shap.png")
            cv2.imwrite(out_path, overlay)
            saved += 1



def apply_shap_all(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    tasks: list = None,
    num_images: int = 10,
    background_size: int = 50,
    alpha: float = 0.5
):
    """
    Run SHAP for each task into its own subfolder.
    """
    if tasks is None:
        tasks = [
            "nature_visual",
            "nep_materiality_visual",
            "nep_biological_visual",
            "landscape-type_visual"
        ]

    for task in tasks:
        task_dir = os.path.join(output_dir, task)
        apply_shap(
            multi_model=multi_model,
            device=device,
            data_loader=data_loader,
            output_dir=task_dir,
            num_images=num_images,
            task=task,
            background_size=background_size,
            alpha=alpha
        )

