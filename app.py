import streamlit as st
import pandas as pd
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI-Powered Fashion Recommendation System",
    layout="wide"
)

st.title("AI-Powered Fashion Recommendation System")

# ----------------------------
# ENVIRONMENT DETECTION
# ----------------------------
IS_CLOUD = not os.path.exists("data/images")
IMAGE_DIR = "data/sample_images" if IS_CLOUD else "data/images"

# ----------------------------
# LOAD DATA (SAFE CSV)
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/styles.csv",
        engine="python",
        on_bad_lines="skip"
    )

    df["baseColour"] = df["baseColour"].astype(str).str.strip().str.lower()
    df["season"] = df["season"].astype(str).str.strip().str.lower()
    df["usage"] = df["usage"].astype(str).str.strip().str.lower()
    df["articleType"] = df["articleType"].astype(str).str.strip()

    return df

df = load_data()

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("Preferences")

color = st.sidebar.selectbox("Color", sorted(df["baseColour"].unique()))
season = st.sidebar.selectbox("Season", sorted(df["season"].unique()))
usage = st.sidebar.selectbox("Usage", sorted(df["usage"].unique()))

# ----------------------------
# IMAGE UPLOAD
# ----------------------------
uploaded_image = st.file_uploader(
    "Upload a clothing image (optional)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Item", width=250)

# ----------------------------
# FILTER DATA
# ----------------------------
filtered_df = df[
    df["baseColour"].str.contains(color, na=False) &
    df["season"].str.contains(season, na=False) &
    df["usage"].str.contains(usage, na=False)
]

if filtered_df.empty:
    st.warning("No exact matches found. Showing random similar items.")
    filtered_df = df.sample(6, random_state=42)

# ----------------------------
# DISPLAY RESULTS WITH IMAGES
# ----------------------------
st.subheader("Recommended Items")

cols = st.columns(3)

for idx, (_, row) in enumerate(filtered_df.head(6).iterrows()):
    with cols[idx % 3]:

        img_path = os.path.join(IMAGE_DIR, f"{row['id']}.jpg")

        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.info("Image not available in demo")

        st.markdown(f"""
        **Article Type:** {row['articleType']}  
        **Color:** {row['baseColour'].title()}  
        **Season:** {row['season'].title()}  
        **Usage:** {row['usage'].title()}
        """)
