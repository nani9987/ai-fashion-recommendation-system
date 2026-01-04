import streamlit as st
from PIL import Image

from src.data_loader import load_dataset, apply_filters
from src.recommender import get_recommendations
from src.utils import get_image_path

st.set_page_config(page_title="AI Fashion Recommender", layout="wide")

st.title("AI-Powered Fashion Recommendation System")

# Load data
df = load_dataset()

# Sidebar filters
st.sidebar.header("Preferences")

color = st.sidebar.selectbox(
    "Color",
    ["All"] + sorted(df["baseColour"].dropna().unique().tolist())
)

season = st.sidebar.selectbox(
    "Season",
    ["All"] + sorted(df["season"].dropna().unique().tolist())
)

usage = st.sidebar.selectbox(
    "Usage",
    ["All"] + sorted(df["usage"].dropna().unique().tolist())
)

filtered_df = apply_filters(df, color, season, usage)

# Image upload
uploaded = st.file_uploader("Upload a clothing image (optional)", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Item", width=250)

# Recommendations
st.subheader("Recommended Items")

recs = get_recommendations(filtered_df)

if len(recs) == 0:
    st.warning("No items found for selected filters.")
else:
    cols = st.columns(3)
    for idx, (_, row) in enumerate(recs.iterrows()):
        col = cols[idx % 3]
        img_path = get_image_path(row["id"])

        if img_path:
            col.image(img_path, width=200)
            col.markdown(f"**{row['articleType']}**")
            col.write(f"Color: {row['baseColour']}")
            col.write(f"Season: {row['season']}")
            col.write(f"Usage: {row['usage']}")
