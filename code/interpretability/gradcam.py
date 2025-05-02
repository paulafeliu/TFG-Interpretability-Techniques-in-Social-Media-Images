"""
gradcam.py
----------
Provides an implementation of GradCAM for visualizing CNN predictions.
"""
import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: The complete model (e.g. MultiTaskModel)
        target_layer: The convolutional layer to hook.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: Tensor of shape (1, C, H, W)
        target_class: Integer index of the target class (default uses predicted class)
        Returns:
            Normalized CAM as a numpy array of shape (H, W)
        """
        outputs = self.model(input_tensor)
        # Select the "nature_visual" output for demonstration (outputs[0])
        score = outputs[0]
        if target_class is None:
            target_class = score.argmax(dim=1).item()
        one_hot = torch.zeros_like(score)
        one_hot[0, target_class] = 1
        self.model.zero_grad()
        score.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.cpu().squeeze().numpy()



def run_gradcam(model, test_dataset, device):
    gradcam_output_dir = os.path.join(OUTPUT_DIR, "grad_cam")
    os.makedirs(gradcam_output_dir, exist_ok=True)

    for img_idx in range(15):
        sample_image, sample_image_name, sample_labels = test_dataset[img_idx]
        input_tensor = sample_image.unsqueeze(0).to(device)
        # Assuming the target layer (for ResNet-based backbones) is layer4
        target_layer = model.backbone.backbone.layer4
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)
        cam_resized = cv2.resize(cam, (224, 224))
        
        original_img = denormalize_image(sample_image)
        
        plt.imshow(original_img)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title("GradCAM - Nature Visual")
        plt.axis('off')

        gradcam_path = os.path.join(OUTPUT_DIR, "grad_cam", f"{sample_image_name}_gradcam.png")
        plt.savefig(gradcam_path, bbox_inches='tight')
        plt.close()
        print(f"GradCAM output saved to {gradcam_path}")