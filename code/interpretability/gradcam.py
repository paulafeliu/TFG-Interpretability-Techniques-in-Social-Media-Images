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
