import os
from src.config import IMAGE_DIR

def get_image_path(item_id):
    path = os.path.join(IMAGE_DIR, f"{item_id}.jpg")
    return path if os.path.exists(path) else None
