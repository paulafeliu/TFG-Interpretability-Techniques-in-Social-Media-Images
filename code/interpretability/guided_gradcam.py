"""
guided backpropagation.py 

"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.wrappers import SingleOutputWrapper
from interpretability.gradcam_backprop import GradCAMBackprop
from utils.denormalize_image import denormalize_image


class GuidedBackprop:
    """
    Produce guided backpropagation saliency maps.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.forward_relu_outputs = []
        self._register_hooks()

    def _register_hooks(self):
        # Hook every ReLU: save forward output, modify backward gradient
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # Store ReLU output for backward pass
        self.forward_relu_outputs.append(output)

    def _backward_hook(self, module, grad_in, grad_out):
        # Only allow positive gradients where forward activation > 0
        forward_output = self.forward_relu_outputs.pop()
        positive_mask = (forward_output > 0).float()
        guided_grad = torch.clamp(grad_out[0], min=0.0) * positive_mask
        return (guided_grad,)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        input_tensor: 1×C×H×W with requires_grad=True
        returns: C×H×W guided backprop gradients
        """
        # Ensure gradients on input
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        # Zero existing gradients
        self.model.zero_grad()
        # Forward + backward
        output = self.model(input_tensor)
        loss = output[0, target_class]
        loss.backward(retain_graph=True)
        # Extract and return gradients wrt input
        grad = input_tensor.grad.data[0]
        return grad.cpu().numpy()


def apply_guided_gradcam(
    multi_model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    output_dir: str,
    num_images: int = 10,
    task: str = "nature_visual",
    target_layer: str = "backbone.backbone.layer4",
    alpha: float = 0.5
):
    """
    Runs Guided Grad-CAM on the first num_images from data_loader for a given task.
    Saves the fused maps (guided Grad-CAM) as overlaid images in output_dir.
    """
    # Map task name to output index
    task_to_index = {
        "nature_visual": 0,
        "nep_materiality_visual": 1,
        "nep_biological_visual": 2,
        "landscape-type_visual": 3
    }
    # Resolve index
    idx = task_to_index.get(task) if isinstance(task, str) else task
    if idx is None:
        raise ValueError(f"Unknown task '{task}'")

    # Wrap model for single-output
    wrapper = SingleOutputWrapper(multi_model, output_index=idx)
    wrapper.to(device).eval()

    # Initialize Grad-CAM and Guided Backprop
    gradcam = GradCAMBackprop(wrapper, target_layer=target_layer)    
    guided_bp = GuidedBackprop(wrapper)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        # Ensure input gradients are available
        imgs.requires_grad_()
        # Forward to get predictions
        outs = wrapper(imgs)
        preds = outs.argmax(dim=1).cpu().tolist()

        for b, name in enumerate(names):
            if saved >= num_images:
                return

            inp = imgs[b].unsqueeze(0)        # 1×C×H×W
            cls = preds[b]

            # 1) Compute small Grad-CAM
            cam_small = gradcam.generate(inp, cls)  # h×w
            # Resize cam to original image size
            orig_img = denormalize_image(imgs[b].detach()) 
            H, W, _ = orig_img.shape
            cam = cv2.resize(cam_small, (W, H))      # H×W

            # 2) Compute guided backprop gradients
            gb = guided_bp.generate(inp, cls)        # C×h×w
            gb = np.moveaxis(gb, 0, 2)               # h×w×C
            gb = cv2.resize(gb, (W, H))              # H×W×C
            gb = np.moveaxis(gb, 2, 0)               # C×H×W

            # 3) Fuse: weight each guided channel by the cam mask and sum
            guided_cam = np.sum(gb * cam[None, :, :], axis=0)
            # Normalize fused map
            guided_cam = (guided_cam - guided_cam.min()) / (guided_cam.max() - guided_cam.min() + 1e-8)

            # 4) Overlay
            heat = np.uint8(255 * guided_cam)
            heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatc, alpha,
                                      np.uint8(255 * orig_img), 1 - alpha, 0)

            # Save
            out_path = os.path.join(output_dir,
                                    f"{name}_{task}_guided_gradcam.png")
            cv2.imwrite(out_path, overlay)
            saved += 1



def apply_guided_gradcam_all(
    multi_model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    output_dir: str,
    tasks: list = None,
    num_images: int = 10,
    target_layer: str = "backbone.backbone.layer4",
    alpha: float = 0.5
):
    """
    Run guided Grad-CAM for each task in `tasks`, saving num_images overlays
    per task into subdirectories under output_dir.
    """
    if tasks is None:
        tasks = ["nature_visual",
                 "nep_materiality_visual",
                 "nep_biological_visual",
                 "landscape-type_visual"]

    for task in tasks:
        task_out = os.path.join(output_dir, task)
        os.makedirs(task_out, exist_ok=True)
        apply_guided_gradcam(
            multi_model=multi_model,
            device=device,
            data_loader=data_loader,
            output_dir=task_out,
            num_images=num_images,
            task=task,
            target_layer=target_layer,
            alpha=alpha
        )
