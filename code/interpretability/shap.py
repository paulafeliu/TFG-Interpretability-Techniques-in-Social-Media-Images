"""
shap.py
-------
Provides an implementation of SHAP for visualizing CNN predictions.
"""
import torch
import shap

class SHAPExplainer:
    def __init__(self, model, background, device):
        """
        model: the trained (or wrapped) single-output model.
        background: a batch of background images (tensor) for SHAP.
        device: computation device (cpu or cuda)
        """
        self.model = model
        self.background = background
        self.device = device
        # Use SHAP’s DeepExplainer (suitable for deep networks)
        self.explainer = shap.DeepExplainer(model, background)

    def explain(self, input_tensor, target_task=0):
        """
        Computes SHAP values for the specified task.

        Parameters:
            input_tensor: Tensor of shape (1, C, H, W)
            target_task: Integer index of the output head to explain (default 0, e.g. nature_visual)
            
        Returns:
            A numpy array (H, W) with normalized SHAP values.
        """
        # Disable the strict additivity check by specifying check_additivity=False.
        shap_values = self.explainer.shap_values(input_tensor, check_additivity=False)
        # For a single-output wrapped model, shap_values is a list with one element.
        explanation = shap_values[target_task]
        # The explanation shape is typically (1, C, H, W); average over channels.
        explanation = explanation.mean(axis=1).squeeze()
        # Normalize the explanation to [0, 1] for visualization.
        min_val = explanation.min()
        max_val = explanation.max()
        explanation = (explanation - min_val) / (max_val - min_val + 1e-8)
        return explanation
