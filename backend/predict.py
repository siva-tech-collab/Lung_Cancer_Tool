import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model import LungCNN
from heatmap import generate_heatmap

classes = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# =====================================
# 🔥 LOAD MODEL ONLY ONCE (BIG FIX)
# =====================================
def load_model():
    model = LungCNN()
    model.load_state_dict(torch.load("lung_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()   # ✅ LOAD ONCE

# =====================================
# 🔥 PREDICT FUNCTION
# =====================================
def predict(image_path):

    # ---- LOAD IMAGE ----
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.resize(img, (128,128))

    # ---- PREPROCESS ----
    img = original / 255.0
    img = (img - 0.5) / 0.5

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    img_tensor = torch.tensor(img, dtype=torch.float32)

    # ---- PREDICTION ----
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    result = classes[predicted.item()]
    confidence = confidence.item()

    # =====================================
    # 🔥 HEATMAP (REQUIRES GRADIENT)
    # =====================================
    heatmap = generate_heatmap(model, img_tensor)

    return result, confidence, heatmap, original