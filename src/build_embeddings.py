import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

from src.feature_extractor import extract_features
from src.data_loader import load_dataset
from src.config import IMAGE_DIR, EMBEDDINGS_DIR

def build_embeddings():
    """Generate and save embeddings for all images."""
    print("Loading dataset...")
    df = load_dataset()
    
    embeddings = []
    valid_ids = []
    failed_images = []
    
    start_time = time.time()
    print(f"Processing {len(df)} images...")
    print(f"Images location: {IMAGE_DIR}")
    
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        img_path = IMAGE_DIR / f"{row['id']}.jpg"
        
        if not img_path.exists():
            failed_images.append(row['id'])
            continue
        
        try:
            features = extract_features(str(img_path))
            if features is not None:
                embeddings.append(features)
                valid_ids.append(row["id"])
            else:
                failed_images.append(row['id'])
        except Exception as e:
            print(f"Error processing image {row['id']}: {e}")
            failed_images.append(row['id'])
    
    # Save embeddings
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    if len(embeddings) == 0:
        print("❌ No embeddings generated! Check IMAGE_DIR path and image files.")
        return
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    ids_array = np.array(valid_ids, dtype=np.int64)
    
    np.save(EMBEDDINGS_DIR / "image_embeddings.npy", embeddings_array)
    np.save(EMBEDDINGS_DIR / "image_ids.npy", ids_array)
    
    elapsed = time.time() - start_time
    print(f"✅ Embeddings generated: {len(embeddings)} images in {elapsed:.1f}s")
    print(f"⚠️  Failed to process: {len(failed_images)} images")
    print(f"📁 Saved to: {EMBEDDINGS_DIR}")

if __name__ == "__main__":
    build_embeddings()
