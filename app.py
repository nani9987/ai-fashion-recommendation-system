import streamlit as st
import pandas as pd
import os
import random
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Fashion Recommendation System",
    layout="wide"
)

st.title("AI-Powered Fashion Recommendation System")

# -------------------------------------------------
# BASE PATH (CRITICAL FIX)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "assets" / "sample_images"

# Load available demo images (cloud-safe)
SAMPLE_IMAGES = []
if IMAGE_DIR.exists():
    SAMPLE_IMAGES = [
        img.name for img in IMAGE_DIR.iterdir()
        if img.suffix.lower() == ".jpg"
    ]

# -------------------------------------------------
# LOAD DATA (SAFE CSV HANDLING)
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        BASE_DIR / "data" / "styles.csv",
        engine="python",
        on_bad_lines="skip"
    )

    for col in ["baseColour", "season", "usage", "articleType"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df

df = load_data()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Preferences")

color = st.sidebar.selectbox("Color", sorted(df["baseColour"].unique()))
season = st.sidebar.selectbox("Season", sorted(df["season"].unique()))
usage = st.sidebar.selectbox("Usage", sorted(df["usage"].unique()))

# -------------------------------------------------
# OPTIONAL IMAGE UPLOAD
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
    (df["baseColour"] == color) &
    (df["season"] == season) &
    (df["usage"] == usage)
]

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

        exact_image = IMAGE_DIR / f"{row['id']}.jpg"

        # 1️⃣ Exact image if exists
        if exact_image.exists():
            st.image(str(exact_image), use_column_width=True)

        # 2️⃣ Fallback demo image
        elif SAMPLE_IMAGES:
            fallback = IMAGE_DIR / random.choice(SAMPLE_IMAGES)
            st.image(str(fallback), use_column_width=True)

        # 3️⃣ Absolute fallback
        else:
            st.warning("No demo images available")

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
