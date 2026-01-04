import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from src.feature_extractor import extract_features
from src.config import IMAGE_DIR, CSV_PATH

# Load dataset
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

embeddings = []
valid_ids = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMAGE_DIR, f"{row['id']}.jpg")

    if os.path.exists(img_path):
        features = extract_features(img_path)
        embeddings.append(features)
        valid_ids.append(row["id"])

# Save embeddings
os.makedirs("embeddings", exist_ok=True)

np.save("embeddings/image_embeddings.npy", np.array(embeddings))
np.save("embeddings/image_ids.npy", np.array(valid_ids))

print("✅ Image embeddings generated successfully")
