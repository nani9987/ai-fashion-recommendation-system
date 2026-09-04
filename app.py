import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from src.data_loader import load_dataset
from src.feature_extractor import extract_features
from src.recommender import get_recommendations
from src.config import IMAGE_DIR, EMBEDDINGS_DIR

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Fashion Recommendation System",
    layout="wide"
)

st.title("AI-Powered Fashion Recommendation System")
st.write("Upload a clothing image or browse by filters to get AI-powered visual recommendations!")

# -------------------------------------------------
# LOAD DATA & EMBEDDINGS
# -------------------------------------------------
@st.cache_data
def load_data():
    """Load fashion dataset."""
    df = load_dataset()
    return df

@st.cache_resource
def load_embeddings():
    """Load precomputed embeddings and image IDs."""
    try:
        embeddings = np.load(EMBEDDINGS_DIR / "image_embeddings.npy")
        image_ids = np.load(EMBEDDINGS_DIR / "image_ids.npy")
        st.success("Embeddings loaded successfully!")
        return embeddings, image_ids
    except FileNotFoundError:
        st.error(f"Embeddings not found at {EMBEDDINGS_DIR}")
        st.info("Run this command to generate embeddings:\npython -m src.build_embeddings")
        return None, None

df = load_data()
embeddings, image_ids = load_embeddings()

if embeddings is None:
    st.stop()

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("Search Options")

use_image = st.sidebar.radio("Search by:", ["Image Upload", "Filters"])

color = st.sidebar.selectbox("Color", ["All"] + sorted(df["baseColour"].unique().tolist()))
season = st.sidebar.selectbox("Season", ["All"] + sorted(df["season"].unique().tolist()))
usage = st.sidebar.selectbox("Usage", ["All"] + sorted(df["usage"].unique().tolist()))

num_recommendations = st.sidebar.slider("Number of items", min_value=3, max_value=12, value=6)

st.sidebar.markdown("---")
st.sidebar.info(
    "Image Upload: Uses CNN feature extraction for visual similarity\n"
    "Filters: Metadata-based search (color, season, usage)"
)

# -------------------------------------------------
# IMAGE UPLOAD PATH
# -------------------------------------------------
query_embedding = None
recommendations = None

if use_image == "Image Upload":
    st.subheader("Upload Your Item")
    uploaded_image = st.file_uploader(
        "Choose a clothing image",
        type=["jpg", "png", "jpeg"]
    )
    
    if uploaded_image:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_image, caption="Your Item", use_column_width=True)
        
        with col2:
            st.write("**Extracting features...**")
            try:
                temp_path = Path("/tmp") / uploaded_image.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                query_embedding = extract_features(str(temp_path))
                
                if query_embedding is not None:
                    st.success("Features extracted successfully!")
                    st.metric("Feature Vector Dim", "2048-D ResNet50")
                else:
                    st.error("Could not process image")
            except Exception as e:
                st.error(f"Error processing image: {e}")
else:
    st.subheader("Browse by Filters")

# -------------------------------------------------
# GET RECOMMENDATIONS
# -------------------------------------------------
st.subheader("Recommended Items")

if use_image == "Image Upload" and query_embedding is not None:
    st.write("**Similar items based on visual features (CNN + Cosine Similarity):**")
    recommendations = get_recommendations(
        query_embedding, 
        embeddings, 
        image_ids, 
        df, 
        n=num_recommendations
    )
else:
    recommendations = df.copy()
    
    if color != "All":
        recommendations = recommendations[recommendations["baseColour"] == color.lower()]
    if season != "All":
        recommendations = recommendations[recommendations["season"] == season.lower()]
    if usage != "All":
        recommendations = recommendations[recommendations["usage"] == usage.lower()]
    
    if recommendations.empty:
        st.warning("No exact matches found. Showing random items instead.")
        recommendations = df.sample(min(num_recommendations, len(df)), random_state=42)
    else:
        st.write(f"**Found {len(recommendations)} matching items. Showing top {num_recommendations}:**")
        recommendations = recommendations.head(num_recommendations)

# -------------------------------------------------
# DISPLAY RECOMMENDATIONS
# -------------------------------------------------
if not recommendations.empty:
    cols = st.columns(3)
    
    for idx, (_, row) in enumerate(recommendations.iterrows()):
        with cols[idx % 3]:
            img_path = IMAGE_DIR / f"{row['id']}.jpg"
            
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
            else:
                st.info(f"Image ID: {row['id']}")
            
            st.markdown(
                f"""
                **Article:** {row.get('articleType', 'N/A').title()}  
                **Color:** {row.get('baseColour', 'N/A').title()}  
                **Season:** {row.get('season', 'N/A').title()}  
                **Usage:** {row.get('usage', 'N/A').title()}
                """
            )
            
            if 'similarity_score' in recommendations.columns:
                score = row['similarity_score']
                st.metric("Match Score", f"{score:.1%}")
else:
    st.warning("No items found.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown(
    """
    ### How This Works
    - **Image Upload:** Uses ResNet50 CNN to extract 2048-D feature vectors -> Computes cosine similarity -> Returns top-N similar items
    - **Filters:** Metadata-based search using color, season, and usage attributes
    - **Response Time:** ~500ms-1s per request (depending on CPU/GPU)
    
    **Dataset:** 44K fashion items from Kaggle Fashion Products Dataset
    """
)
