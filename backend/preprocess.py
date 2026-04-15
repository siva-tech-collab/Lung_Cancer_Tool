import cv2
import numpy as np

def preprocess_image(image):
    # Resize
    image = cv2.resize(image, (128,128))

    # Normalize to [0,1]
    image = image / 255.0

    # 🔥 Match training normalization
    image = (image - 0.5) / 0.5

    # Add channel + batch dimension
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    return image