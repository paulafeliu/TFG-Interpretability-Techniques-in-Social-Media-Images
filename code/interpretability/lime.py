"""
lime.py 

"""
import os
import cv2
import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import slic, quickshift
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils.denormalize_image import denormalize_image

class LIMEExplainer:
    def __init__(self, model, device):
        """
        model: A PyTorch model (multi-output or single-output).
        device: torch.device instance.
        """
        self.model = model.to(device)
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()
        # Preprocessing to match model's training transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict_fn(self, images, target_task):
        """
        images: NumPy array of shape (N, H, W, C) with float values in [0,1].
        target_task: Index of the output head to explain (0..3) if model returns a tuple.
        Returns:
            NumPy array of shape (N, num_classes) with class probabilities.
        """
        self.model.eval()
        preds = []
        for img in images:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            tensor_img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor_img)
                # If multi-output model, select the desired head
                if isinstance(outputs, (list, tuple)):
                    out = outputs[target_task]
                else:
                    out = outputs
                # Convert logits to probabilities
                prob = torch.nn.functional.softmax(out, dim=1)
                preds.append(prob.cpu().numpy()[0])
        return np.array(preds)

    def explain(self, image, target_task=0, save_path=None, num_features=10, num_samples=1000):
        """
        Generate and optionally save a LIME explanation.
        image: NumPy array (H, W, C) in [0,1].
        target_task: Which model output to explain.
        save_path: File path to save the overlay image (.png).
        num_features: Number of superpixel features to highlight.
        num_samples: Number of perturbed samples to generate.
        Returns:
            explanation: LIME explanation object.
            temp: Image array with highlighted superpixels.
            mask: Mask array indicating highlighted regions.
        """
        explanation = self.explainer.explain_instance(
            image.astype(np.double),
            classifier_fn=lambda imgs: self.predict_fn(imgs, target_task),
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )

        # build a colored overlay on top of the original image
        overlay = image.copy()      # H×W×3 in [0,1]
        alpha   = 0.5
        cmap    = plt.get_cmap('jet')
        # mask>0 selects the LIME superpixels
        colored = cmap((mask > 0).astype(float))[:, :, :3]
        combined = (1 - alpha) * overlay + alpha * colored

        # Plot & save the blended result
        plt.figure(figsize=(6, 6))
        plt.imshow(combined)
        plt.axis('off')
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        return explanation, combined, mask




def run_lime(model, test_dataset, device, output_dir: str, num_images=10):
    """
    Computes and saves LIME explanations for each task over a subset of test samples.
    - num_images: number of images from the test set to explain.
    """
    

    # Initialize explainer once for the multi-task model
    lime_explainer = LIMEExplainer(model, device)

    # Map task indices to human-readable names
    task_dict = {
        0: "nature_visual",
        1: "nep_materiality_visual",
        2: "nep_biological_visual",
        3: "landscape-type_visual"
    }

    # Ensure base output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Create one subfolder per task
    task_dirs = {}
    for task_idx, task_name in task_dict.items():
        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        task_dirs[task_idx] = task_dir

    # Iterate over a subset of test images
    for idx in range(min(num_images, len(test_dataset))):
        sample_image, sample_image_name, _ = test_dataset[idx]
        image_np = denormalize_image(sample_image)

        # Generate explanations for all tasks
        for task_idx, task_name in task_dict.items():
            save_path = os.path.join(
                task_dirs[task_idx],
                f"{sample_image_name}_lime_{task_name}.png"
            )
            _, temp, mask = lime_explainer.explain(
                image_np,
                target_task=task_idx,
                save_path=save_path,
                num_features=10,
                num_samples=1000
            )




class LIMEExplainerRefined:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        segmentation: str = 'slic',
        n_segments: int = 100,
        compactness: float = 10.0
    ):
        """
        LIME explainer with configurable segmentation and post-processing.
        segmentation: 'slic' or 'quickshift'
        n_segments, compactness: parameters for slic
        """
        self.model = model.to(device).eval()
        self.device = device
        # Choose segmentation function
        if segmentation == 'slic':
            self.segmentation_fn = lambda img: slic(
                img,
                n_segments=n_segments,
                compactness=compactness,
                start_label=1
            )
        else:
            self.segmentation_fn = lambda img: quickshift(
                img,
                kernel_size=4,
                max_dist=200,
                ratio=0.2
            )
        # Preprocessing matches training pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.explainer = lime_image.LimeImageExplainer()

    def predict_fn(self, images: np.ndarray, target_task: int) -> np.ndarray:
        """
        images: array of shape (N, H, W, C) in [0,1]
        """
        self.model.eval()
        batch = []
        for img in images:
            # Convert to tensor
            tensor = self.preprocess(Image.fromarray((img*255).astype(np.uint8)))
            batch.append(tensor)
        batch_tensor = torch.stack(batch).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            # select head
            out = outputs[target_task] if isinstance(outputs, (list, tuple)) else outputs
            probs = torch.softmax(out, dim=1)
        return probs.cpu().numpy()

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_task: int = 0,
        save_path: str = None,
        num_features: int = 10,
        num_samples: int = 1000,
        hide_color: int = None
    ):
        """
        Generate LIME explanation with improved segmentation and overlay.
        hide_color: pixel value to replace hidden superpixels (default mean of image)
        """
        # Denormalize input
        image = denormalize_image(image_tensor.cpu()).astype(np.float32)
        # Optional hide_color as mean
        if hide_color is None:
            hide_color = tuple((image.mean(axis=(0,1))*255).astype(int))
        # LIME explain
        explanation = self.explainer.explain_instance(
            image.astype(np.double),
            classifier_fn=lambda imgs: self.predict_fn(imgs, target_task),
            top_labels=1,
            hide_color=hide_color,
            num_samples=num_samples,
            segmentation_fn=self.segmentation_fn
        )
        idx = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label=idx,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        # Morphological filtering
        hm = (mask>0).astype(np.uint8)*255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        hm = cv2.morphologyEx(hm, cv2.MORPH_OPEN, kernel)
        hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, kernel)
        # Overlay with perceptual colormap
        colored = cv2.applyColorMap(hm, cv2.COLORMAP_VIRIDIS)
        orig_bgr = np.uint8(image*255)[..., ::-1]
        overlay = cv2.addWeighted(colored, 0.6, orig_bgr, 0.4, 0)
        # Draw contours
        contours, _ = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255,255,255), 2)
        # Save
        if save_path:
            cv2.imwrite(save_path, overlay)
        return explanation, overlay, hm


def run_lime_refined(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    output_dir: str,
    num_images: int = 10,
    **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    explainer = LIMEExplainerRefined(model, device, **kwargs)
    task_map = {0:"nature_visual",3:"landscape-type_visual"} # 1:"nep_materiality_visual",2:"nep_biological_visual",
    # Per-image, per-task
    for i in range(min(num_images, len(dataset))):
        img_t, name, _ = dataset[i]
        for t_idx, t_name in task_map.items():
            save_folder = os.path.join(output_dir, t_name)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{name}_lime_{t_name}.png")
            explainer.explain(
                img_t,
                target_task=t_idx,
                save_path=save_path,
                num_features=15,
                num_samples=1500
            )

