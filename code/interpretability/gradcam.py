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
        '''module = self.model
        for name in self.target_layer.split('.'):
            module = getattr(module, name)
        return module'''

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

def apply_gradcam(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    num_images: int = 10,
    task: str = "nature_visual",
    target_layer: str = 'backbone.backbone.layer4',
    alpha: float = 0.5
):
    """
    Generate basic Grad-CAM overlays for one task.
    """
    task_to_index = {
        "nature_visual": 0,
        "nep_materiality_visual": 1,
        "nep_biological_visual": 2,
        "landscape-type_visual": 3
    }
    idx = task_to_index.get(task)
    if idx is None:
        raise ValueError(f"Unknown task '{task}'")

    # Wrap and move to device
    wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
    gradcam = GradCAM(wrapper, target_layer)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        imgs.requires_grad_()
        outs = wrapper(imgs)
        preds = outs.argmax(dim=1).cpu().tolist()

        for b, name in enumerate(names):
            if saved >= num_images:
                return

            inp = imgs[b].unsqueeze(0)    # 1×C×H×W
            cls = preds[b]

            # 1) generate cam mask
            cam_small = gradcam.generate(inp, cls)  # h×w

            # 2) upscale & overlay
            orig = denormalize_image(imgs[b].detach())
            H, W, _ = orig.shape
            cam = cv2.resize(cam_small, (W, H))
            heat = np.uint8(255 * cam)
            heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            #heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[..., ::-1]
            overlay = cv2.addWeighted(heatc, alpha,
                                      np.uint8(255 * orig), 1 - alpha, 0)

            # 3) save
            outfile = os.path.join(output_dir, f"{name}_{task}_gradcam.png")
            cv2.imwrite(outfile, overlay)
            saved += 1


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
    Wrapper to run basic Grad-CAM over all specified tasks,
    each into its own subfolder.
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
        apply_gradcam(
            multi_model=multi_model,
            device=device,
            data_loader=data_loader,
            output_dir=task_dir,
            num_images=num_images,
            task=task,
            target_layer=target_layer,
            alpha=alpha
        )



def apply_gradcam_pair(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader,
    output_dir: str,
    task_pair: tuple = ("nature_visual", "landscape-type_visual"),
    target_layer: str = 'backbone.backbone.layer4',
    alpha: float = 0.5
):
    """
    Generate a single Grad-CAM overlay with TWO tasks, each encoded in its own RGB channel.
    Nature → RED, Landscape → GREEN, overlap → YELLOW.
    """
    task_to_index = {
        "nature_visual": 0,
        "nep_materiality_visual": 1,
        "nep_biological_visual": 2,
        "landscape-type_visual": 3
    }

    # build a GradCAM instance for each task
    cams = []
    for task in task_pair:
        idx = task_to_index.get(task)
        if idx is None:
            raise ValueError(f"Unknown task '{task}'")
        wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
        cams.append(GradCAM(wrapper, target_layer))

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        imgs.requires_grad_()

        # get each task's predicted class
        preds = []
        for cam in cams:
            out = cam.model(imgs)
            preds.append(out.argmax(dim=1).cpu().tolist())

        for b, name in enumerate(names):
            inp  = imgs[b].unsqueeze(0)
            orig = denormalize_image(imgs[b].detach())  # float32 H×W×3 in [0,1]
            H, W, _ = orig.shape

            # 1) generate each CAM (float H×W in [0,1])
            cam_maps = []
            for cam_obj, pred_list in zip(cams, preds):
                cm = cam_obj.generate(inp, pred_list[b])
                cm = cv2.resize(cm, (W, H))
                cam_maps.append(cm)

            # 2) build RGB heat: red=cam0, green=cam1, blue=0
            heat_rgb = np.zeros((H, W, 3), dtype=np.float32)
            heat_rgb[..., 0] = cam_maps[0]   # RED channel for task_pair[0]
            heat_rgb[..., 1] = cam_maps[1]   # GREEN channel for task_pair[1]
            # leave heat_rgb[...,2] == 0 (blue)

            # 3) to uint8 & optionally blur
            heat_rgb = np.clip(heat_rgb * 255, 0, 255).astype(np.uint8)
            heat_rgb = cv2.GaussianBlur(heat_rgb, (7, 7), sigmaX=0)

            # 4) overlay onto original
            orig_uint8 = (orig * 255).astype(np.uint8)
            overlay   = cv2.addWeighted(heat_rgb, alpha, orig_uint8, 1 - alpha, 0)

            # 5) save
            outname = f"{name}_{task_pair[0]}_{task_pair[1]}_rgbcam.png"
            cv2.imwrite(os.path.join(output_dir, outname), overlay)
            saved += 1