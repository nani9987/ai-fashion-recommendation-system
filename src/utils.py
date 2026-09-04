from pathlib import Path
from src.config import IMAGE_DIR

def get_image_path(item_id):
    """Get image path for item ID if it exists."""
    try:
        path = IMAGE_DIR / f"{item_id}.jpg"
        return path if path.exists() else None
    except (TypeError, ValueError):
        return None

def validate_image(item_id):
    """Check if image file exists and is readable."""
    path = get_image_path(item_id)
    return path is not None
