import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap(model, image_tensor):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.conv2

    # 🔥 FIX 1: register BOTH hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # 🔥 FIX 2: enable gradients properly
    image_tensor.requires_grad = True

    # Forward
    output = model(image_tensor)
    pred_class = output.argmax()

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    # 🔥 FIX 3: safety check (prevents crash)
    if len(gradients) == 0 or len(activations) == 0:
        raise ValueError("Hooks did not capture gradients/activations")

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2,3), keepdim=True)

    heatmap = torch.sum(weights * acts, dim=1).squeeze()
    heatmap = torch.relu(heatmap)

    heatmap = heatmap.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (128,128))

    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap