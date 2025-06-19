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
        """
        model:        your SingleOutputWrapper around the multi-task net
        target_layer: e.g. "backbone.backbone.features.8"
        """
        self.model = model
        self.activations = None
        self.gradients   = None

        # 1) unwrap the wrapper if present
        module = self.model
        if hasattr(module, 'model'):
            module = module.model

        # 2) walk the dotted path, treating numeric segments as Sequential indices
        for name in target_layer.split('.'):
            if name.isdigit() and hasattr(module, '__getitem__'):
                module = module[int(name)]
            else:
                module = getattr(module, name)

        # 3) register our hooks on that conv layer
        module.register_forward_hook(self._save_activation)
        module.register_backward_hook(self._save_gradient)

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
        cam = torch.zeros(activ.shape[1:], dtype=torch.float32, device=activ.device)
        for i, w in enumerate(weights):
            cam += w * activ[i]

        cam = F.relu(cam).cpu().numpy()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def apply_gradcam_backprop_all(
    multi_model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    output_dir: str,
    tasks: Union[str, List[str]] = "nature_visual",
    num_images: int = 10,
    target_layer: str = "backbone.backbone.layer4",
    alpha: float = 0.5
):
    """
    Runs Grad‐CAM backprop for one or more tasks, taking the first num_images
    from data_loader for each task, annotating and saving overlays under
    subdirectories of output_dir.
    """
    # mapping task -> index
    task_to_index = {
        "nature_visual":          0,
        "nep_materiality_visual": 1,
        "nep_biological_visual":  2,
        "landscape-type_visual":  3
    }

    # normalize tasks to a list
    if isinstance(tasks, str):
        tasks = [tasks]
    for t in tasks:
        if t not in task_to_index:
            raise ValueError(f"Unknown task '{t}'")

    os.makedirs(output_dir, exist_ok=True)
    saved = {t: 0 for t in tasks}

    # get inverse label‐maps once
    raw_ds = data_loader.dataset
    if isinstance(raw_ds, torch.utils.data.Subset):
        raw_ds = raw_ds.dataset
    inv_maps = {
        t: {v: k for k, v in raw_ds.label_maps[t].items()}
        for t in tasks
    }

    for task in tasks:
        idx = task_to_index[task]
        task_dir = os.path.join(output_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        wrapper = SingleOutputWrapper(multi_model, output_index=idx).to(device).eval()
        gradcam = GradCAMBackprop(wrapper, target_layer=target_layer)

        for imgs, names, labels in data_loader:
            imgs = imgs.to(device)
            outs = wrapper(imgs)                   # [B×K]
            preds = outs.argmax(dim=1).cpu()       # Tensor[B]

            for b, name in enumerate(names):
                if saved[task] >= num_images:
                    break

                inp       = imgs[b].unsqueeze(0)          # 1×C×H×W
                cls       = preds[b].item()               # predicted int
                true_cls  = labels[task][b].item()        # ground‐truth int
                pred_str  = inv_maps[task][cls]
                true_str  = inv_maps[task][true_cls]

                # compute and resize CAM
                cam = gradcam.generate(inp, cls)          # h×w in [0..1]
                orig_h, orig_w, _ = denormalize_image(imgs[b]).shape
                cam = cv2.resize(cam, (orig_w, orig_h))

                # overlay
                img_np = denormalize_image(imgs[b])       # H×W×3 in [0..1]
                heat   = np.uint8(255 * cam)
                heatc  = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[..., ::-1]
                over   = cv2.addWeighted(
                    heatc, alpha,
                    np.uint8(255 * img_np), 1 - alpha,
                    0
                )

                # annotate
                text = f"PRED: {pred_str}   TRUE: {true_str}"
                cv2.putText(
                    over, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), thickness=2,
                    lineType=cv2.LINE_AA
                )

                out_path = os.path.join(task_dir, f"{name}_{task}_pred{cls}.png")
                cv2.imwrite(out_path, over)
                saved[task] += 1

            if saved[task] >= num_images:
                break
