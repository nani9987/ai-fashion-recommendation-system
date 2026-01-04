import streamlit as st
import pandas as pd
import os
import random

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Fashion Recommendation System",
    layout="wide"
)

st.title("AI-Powered Fashion Recommendation System")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
IMAGE_DIR = "assets/sample_images"

# Load available demo images
SAMPLE_IMAGES = []
if os.path.exists(IMAGE_DIR):
    SAMPLE_IMAGES = [
        img for img in os.listdir(IMAGE_DIR)
        if img.lower().endswith(".jpg")
    ]

# -------------------------------------------------
# LOAD DATA (SAFE CSV HANDLING)
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/styles.csv",
        engine="python",
        on_bad_lines="skip"
    )

    # Normalize text columns
    for col in ["baseColour", "season", "usage", "articleType"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df

df = load_data()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Preferences")

color = st.sidebar.selectbox(
    "Color", sorted(df["baseColour"].unique())
)
season = st.sidebar.selectbox(
    "Season", sorted(df["season"].unique())
)
usage = st.sidebar.selectbox(
    "Usage", sorted(df["usage"].unique())
)

# -------------------------------------------------
# IMAGE UPLOAD (OPTIONAL)
# -------------------------------------------------
uploaded_image = st.file_uploader(
    "Upload a clothing image (optional)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Item", width=250)

# -------------------------------------------------
# FILTER DATA
# -------------------------------------------------
filtered_df = df[
    df["baseColour"].str.contains(color, na=False) &
    df["season"].str.contains(season, na=False) &
    df["usage"].str.contains(usage, na=False)
]

# Fallback if no matches
if filtered_df.empty:
    st.warning("No exact matches found. Showing similar items instead.")
    filtered_df = df.sample(6, random_state=42)

# -------------------------------------------------
# DISPLAY RECOMMENDATIONS
# -------------------------------------------------
st.subheader("Recommended Items")

cols = st.columns(3)

for idx, (_, row) in enumerate(filtered_df.head(6).iterrows()):
    with cols[idx % 3]:

        # Try exact image by ID
        img_path = os.path.join(IMAGE_DIR, f"{row['id']}.jpg")

        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)

        # Fallback to random demo image
        elif SAMPLE_IMAGES:
            fallback_img = random.choice(SAMPLE_IMAGES)
            fallback_path = os.path.join(IMAGE_DIR, fallback_img)
            st.image(fallback_path, use_column_width=True)

        else:
            st.info("Image not available")

        st.markdown(
            f"""
            **Article Type:** {row['articleType'].title()}  
            **Color:** {row['baseColour'].title()}  
            **Season:** {row['season'].title()}  
            **Usage:** {row['usage'].title()}
            """
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption(
    "Cloud demo uses a sampled image set. "
    "Full CNN-based recommendations run locally."
)
