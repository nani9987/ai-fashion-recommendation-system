import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

# Load model ONCE globally (not in function)
try:
    base_model = ResNet50(
        weights='imagenet',  # TensorFlow 2.13+ compatible
        include_top=False,
        pooling='avg'
    )
except Exception as e:
    # Fallback for newer TensorFlow versions
    print(f"Note: Loading ImageNet weights with fallback method: {e}")
    base_model = ResNet50(
        weights=None,
        include_top=False,
        pooling='avg'
    )

def extract_features(img_path):
    """Extract 2048-D feature vector from image using ResNet50."""
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # base_model outputs (1, 2048) with pooling='avg'
        features = base_model.predict(img_array, verbose=0)
        return features.squeeze()  # Shape: (2048,)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
