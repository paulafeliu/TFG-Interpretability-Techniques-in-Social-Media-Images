import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from models.wrappers import SingleOutputWrapper
from data_processing.dataset import MultiTaskImageDataset
from utils.denormalize_image import denormalize_image


class GradCAMBackprop:
    """
    Hook the specified conv‐layer of a single‐output model,
    compute Grad‐CAM for a target class.
    """
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.activations = None
        self.gradients   = None

        # register hooks
        for name, module in self.model.named_modules():
            if name == target_layer:
                module.register_forward_hook(self._save_activation)
                module.register_backward_hook(self._save_gradient)
                break
        else:
            raise ValueError(f"Layer '{target_layer}' not found in model.")

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] is gradients w.r.t. forward‐output
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        x: 1×C×H×W input tensor
        class_idx: integer class index
        returns: H×W heatmap in [0..1]
        """
        self.model.zero_grad()
        out = self.model(x)                   # shape [1×K]
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        activ = self.activations[0]           # C×h×w
        grads = self.gradients[0]             # C×h×w

        # global‐avg pool over spatial dims → C weights
        weights = grads.view(grads.size(0), -1).mean(dim=1)

        # weighted sum of activations
        #cam = torch.zeros(activ.shape[1:], dtype=torch.float32)
        cam = torch.zeros(activ.shape[1:], dtype=torch.float32, device=activ.device)
        for i, w in enumerate(weights):
            cam += w * activ[i]

        cam = F.relu(cam).cpu().numpy()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def apply_gradcam_backprop(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    output_dir: str,
    num_images: int = 10,
    task: str = "nature_visual",
    target_layer: str = "backbone.backbone.layer4",
    alpha: float = 0.5
):
    """
    Runs Grad‐CAM on the first num_images from data_loader
    for the given task, and writes overlays to output_dir.
    """
    #os.makedirs(output_dir, exist_ok=True)

    # wrap your multi‐task model into a single‐head for `task`
    #wrapper = SingleOutputWrapper(multi_model, output_index=task)

    # mapear nombre de tarea a índice
    task_to_index = {
        "nature_visual":                 0,
        "nep_materiality_visual":        1,
        "nep_biological_visual":         2,
        "landscape-type_visual":         3
    }
    if isinstance(task, str):
        idx = task_to_index.get(task)
        if idx is None:
            raise ValueError(f"Tarea desconocida '{task}'")
    else:
        idx = task

    wrapper = SingleOutputWrapper(multi_model, output_index=idx)
    wrapper.to(device).eval()

    gradcam = GradCAMBackprop(wrapper, target_layer=target_layer)

    saved = 0
    for imgs, names, _ in data_loader:
        imgs = imgs.to(device)
        # forward through the single‐head wrapper
        outs  = wrapper(imgs)                       # [B×K]
        preds = outs.argmax(dim=1).cpu().tolist()   # [B]

        for b, name in enumerate(names):
            if saved >= num_images:
                print(f"Saved {saved} Grad‐CAMs → done.")
                return

            inp = imgs[b].unsqueeze(0)              # 1×C×H×W
            cls = preds[b]

            # compute heatmap
            #cam = gradcam.generate(inp, cls)       # H×W in [0..1]

            # compute heatmap (tamaño pequeño) y reescalar al tamaño de entrada
            cam = gradcam.generate(inp, cls)       # cam: h×w en [0..1]
            # redimensionar al tamaño original de la imagen
            orig_h, orig_w, _ = denormalize_image(imgs[b]).shape
            cam = cv2.resize(cam, (orig_w, orig_h))  # ahora cam es H×W
 
            # de‐normalize imagen y superponer
            img_np = denormalize_image(imgs[b])    # H×W×3 en [0..1]
            heat  = np.uint8(255 * cam)
            heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[..., ::-1]
            over  = cv2.addWeighted(heatc, alpha,
                                    np.uint8(255 * img_np), 1 - alpha, 0)

            out_p = os.path.join(output_dir,
                                 f"{name}_{task}_pred{cls}.png")
            cv2.imwrite(out_p, over)
            saved += 1

    print(f"Exhausted data – saved {saved} Grad‐CAMs.")


def apply_gradcam_backprop_all(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    output_dir: str,
    tasks: list = None,
    num_images: int = 10,
    target_layer: str = "backbone.backbone.layer4",
    alpha: float = 0.5
):
    """
    Run GradCAM for each task in `tasks`, saving num_images overlays
    per task into subdirs of output_dir.
    """
    if tasks is None:
        tasks = [
            "nature_visual",
            "nep_materiality_visual",
            "nep_biological_visual",
            "landscape-type_visual"
        ]

    for task in tasks:
        task_out = os.path.join(output_dir, task)
        os.makedirs(task_out, exist_ok=True)
        print(f"\n→ Generating {num_images} Grad‑CAMs for '{task}', saving to:\n   {task_out}")
        apply_gradcam_backprop(
            multi_model=multi_model,
            device=device,
            data_loader=data_loader,
            output_dir=task_out,
            num_images=num_images,
            task=task,
            target_layer=target_layer,
            alpha=alpha
        )
