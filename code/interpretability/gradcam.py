"""
gradcam.py 

"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models.wrappers import SingleOutputWrapper
from utils.denormalize_image import denormalize_image

class GradCAM:
    """
    Basic Grad-CAM implementation.
    """
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        model: a single-output wrapper around your multi-task model
        target_layer: e.g. "backbone.backbone.layer4"
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _resolve_module(self):
        # 1) unwrap SingleOutputWrapper (it stores your real model in .model)
        module = self.model
        if hasattr(module, 'model'):
            module = module.model

        # 2) follow the dotted path, treating numeric names as indices
        for name in self.target_layer.split('.'):
            if name.isdigit() and hasattr(module, '__getitem__'):
                module = module[int(name)]
            else:
                module = getattr(module, name)
        return module

    def _register_hooks(self):
        layer = self._resolve_module()
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, outp):
        self.activations = outp.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        input_tensor: 1×C×H×W, requires_grad_(True)
        Returns a normalized H×W heatmap in [0,1].
        """ 
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        # Forward + backward
        output = self.model(input_tensor)              # 1×num_classes
        loss = output[0, target_class]
        loss.backward()

        # Weights: global avg pool of gradients
        grads = self.gradients[0]                      # C×h×w
        weights = grads.view(grads.size(0), -1).mean(dim=1)  # C

        # Weighted sum of activations
        activations = self.activations[0]              # C×h×w
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU & normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam



def apply_gradcam_all(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    tasks: list = None,
    num_images: int = 10,
    target_layer: str = 'backbone.backbone.layer4',
    alpha: float = 0.5
):
    """
    Generate and save Grad-CAM overlays for one or multiple tasks.
    """
    # Mapping task → output index
    task_to_index = {
        "nature_visual": 0,
        "nep_materiality_visual": 1,
        "nep_biological_visual": 2,
        "landscape-type_visual": 3
    }
    if tasks is None:
        tasks = list(task_to_index.keys())

    os.makedirs(output_dir, exist_ok=True)
    saved = {t: 0 for t in tasks}

    # For each task, we wrap the multi_model and create its GradCAM
    for task in tasks:
        idx = task_to_index.get(task)
        if idx is None:
            raise ValueError(f"Unknown task '{task}'")
        subdir = os.path.join(output_dir, task)
        os.makedirs(subdir, exist_ok=True)

        wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
        gradcam = GradCAM(wrapper, target_layer)

        # Loop through the loader until reaching num_images
        for imgs, names, _ in data_loader:
            imgs = imgs.to(device)
            imgs.requires_grad_()
            outs = wrapper(imgs)
            preds = outs.argmax(dim=1).cpu().tolist()

            for b, name in enumerate(names):
                if saved[task] >= num_images:
                    break

                inp = imgs[b].unsqueeze(0)
                cls = preds[b]

                cam_small = gradcam.generate(inp, cls)
                orig = denormalize_image(imgs[b].detach())
                H, W, _ = orig.shape
                cam = cv2.resize(cam_small, (W, H))
                heat = np.uint8(255 * cam)
                heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(heatc, alpha, np.uint8(255 * orig), 1 - alpha, 0)

                outpath = os.path.join(subdir, f"{name}_{task}_gradcam.png")
                cv2.imwrite(outpath, overlay)
                saved[task] += 1

            if saved[task] >= num_images:
                break