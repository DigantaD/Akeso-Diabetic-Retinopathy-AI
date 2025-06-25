import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[:, class_idx]
        loss.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        for i in range(activations.shape[0]):
            activations[i, ...] *= pooled_grads[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max()
        return heatmap.numpy()

    def overlay(self, heatmap, original_img, alpha=0.4):
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
        return overlay_img

class GradCAMSegmentation:
    def __init__(self, model, target_layer, device):
        self.model = model.to(device)
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_channel=0):
        """
        input_tensor: B x 3 x H x W
        target_channel: which lesion mask (0=MA, ..., 4=OD)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        self.model.zero_grad()
        output = self.model(input_tensor)  # B x 5 x H x W
        loss = output[:, target_channel].mean()
        loss.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)  # C x H x W

        for i in range(activations.shape[0]):
            activations[i, ...] *= pooled_grads[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.numpy()

    def overlay(self, heatmap, original_img, alpha=0.4):
        """
        original_img: H x W x 3 (uint8)
        """
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
        return overlay_img