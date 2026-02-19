"""
Shield-Ryzen V2 — GradCAM Explainability (TASK 10.2)
====================================================
Generates visual explanations for model decisions.
Highlights regions contributing to "FAKE" classification.

Note: Requires PyTorch model access. Returns placeholder if running
purely on ONNX Runtime without torch gradient support.

Developer: Inayat Hussain | AMD Slingshot 2026
Part 10 of 14 — HUD & Explainability
"""

import cv2
import numpy as np
import logging

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class GradCAMExplainer:
    def __init__(self, model=None, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.enabled = (HAS_TORCH and model is not None)
        
        if self.enabled:
            # Register hooks
            # This assumes model is a PyTorch nn.Module
            try:
                target_layer.register_forward_hook(self.save_activation)
                target_layer.register_backward_hook(self.save_gradient)
            except Exception as e:
                logging.warn(f"GradCAM init failed: {e}")
                self.enabled = False

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class_idx):
        """
        Generate GradCAM heatmap.
        input_tensor: (1, 3, H, W)
        """
        if not self.enabled:
            return None # Fallback or disabled

        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class_idx] = 1
        output.backward(gradient=one_hot)
        
        # Compute weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()

    def process_frame_stub(self, frame_crop):
        """
        Simulated Heatmap for UI testing when PyTorch model unavailable.
        Uses high-frequency entropy as proxy for 'attention'.
        """
        if frame_crop is None: return None
        gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        
        # Edge density as heatmap
        edges = cv2.Canny(gray, 100, 200)
        heatmap = cv2.GaussianBlur(edges, (21, 21), 0)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Colorize
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        return heatmap_color
