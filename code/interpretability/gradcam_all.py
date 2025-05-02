# interpretability/gradcam.py
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.wrappers import SingleOutputWrapper
from utils.denormalize_image import denormalize_image

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        model: Un modelo que devuelve un solo tensor de logits (ya envuelto por SingleOutputWrapper).
        target_layer: La capa convolucional donde enganchar los hooks.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Registrar hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] son los gradientes respecto a la salida forward
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Genera la máscara CAM normalizada en [0..1].
        - input_tensor: (1, C, H, W)
        - target_class: índice de la clase; por defecto usa la predicción.
        """
        # Forward + backward
        outputs = self.model(input_tensor)            # shape [1, num_classes]
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        score = outputs[0, target_class]
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1

        #self.model.zero_grad()
        #score.backward(gradient=one_hot, retain_graph=True)
        self.model.zero_grad()
        outputs.backward(gradient=one_hot, retain_graph=True)

        # Pesos: average pooling de gradientes sobre h×w
        grads = self.gradients[0]                     # (C, h, w)
        weights = grads.view(grads.size(0), -1).mean(dim=1)

        # Combinar pesos con activations
        activ = self.activations[0]                   # (C, h, w)
        #cam = torch.zeros(activ.shape[1:], dtype=torch.float32)
        cam = torch.zeros(activ.shape[1:], dtype=torch.float32, device=activ.device)
        
        for i, w in enumerate(weights):
            cam += w * activ[i]
        cam = torch.relu(cam).cpu().numpy()

        # Normalizar a [0..1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

def run_gradcam_all(
    multi_model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    output_dir: str,
    num_images: int = 15,
    alpha: float = 0.5
):
    """
    Para cada una de las 4 tareas del modelo multitarea, genera Grad-CAMs
    para los primeros `num_images` y guarda las superposiciones en:
        output_dir/grad_cam/<task_name>/
    """
    # Mapa índice → nombre de task
    task_dict = {
        0: "nature_visual",
        1: "nep_materiality_visual",
        2: "nep_biological_visual",
        3: "landscape-type_visual"
    }

    base_out = os.path.join(output_dir, "grad_cam")
    os.makedirs(base_out, exist_ok=True)

    # Capa convolucional objetivo en tu CustomBackbone
    # Asegúrate de usar la misma ruta que en named_modules(): 
    #   model.backbone.backbone.layer4
    # Aquí extraemos la capa directamente
    target_layer = multi_model.backbone.backbone.layer4

    for task_idx, task_name in task_dict.items():
        task_dir = os.path.join(base_out, task_name)
        os.makedirs(task_dir, exist_ok=True)

        print(f"\n→ Generando Grad-CAM para '{task_name}' (índice {task_idx}) → {task_dir}")

        # Envolver el modelo para que devuelva solo la salida task_idx
        single_model = SingleOutputWrapper(multi_model, output_index=task_idx)
        single_model.to(device).eval()

        gradcam = GradCAM(single_model, target_layer)

        saved = 0
        for img_idx in range(len(test_dataset)):
            if saved >= num_images:
                break

            img, img_name, _ = test_dataset[img_idx]
            inp = img.unsqueeze(0).to(device)

            # Generar CAM y resize
            cam = gradcam.generate_cam(inp)
            cam_resized = cv2.resize(cam, (224, 224))

            # Desenormalizar imagen original
            orig = denormalize_image(img)  # H×W×3 en [0..1]

            # Crear overlay
            # 1) Genera el heatmap coloreado
            heat  = np.uint8(255 * cam_resized)
            heatc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[..., ::-1]
        
            # 2) Prepara la imagen original en uint8
            orig_uint8 = np.uint8(255 * orig)  # (H, W, 3) o (h', w', 3)
        
            # 3) Asegúrate de que orig y heatc tienen la misma resolución
            h, w = heatc.shape[:2]
            if orig_uint8.shape[0] != h or orig_uint8.shape[1] != w:
                orig_uint8 = cv2.resize(orig_uint8, (w, h))
        
            # 4) Overlay
            over = cv2.addWeighted(heatc, alpha,
                                   orig_uint8, 1 - alpha, 0)

            out_path = os.path.join(task_dir, f"{img_name}_{task_name}.png")
            cv2.imwrite(out_path, over)
            saved += 1

        print(f"   → Guardadas {saved} imágenes.")
