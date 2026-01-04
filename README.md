# AI-Powered Fashion Recommendation System 👕👖👟

## 📌 Project Overview
This project is an **AI-powered, image-first fashion recommendation system** that suggests visually similar clothing items and automatically completes outfits (topwear, bottomwear, footwear) based on a user-uploaded image and selected preferences.

Unlike traditional e-commerce platforms that rely heavily on **text search, brand information, or purchase history**, this system focuses on **computer vision and visual similarity**, enabling style-based recommendations even without prior user data.

---

## 🎯 Key Features
- 📷 Image-based fashion recommendations using CNN embeddings (ResNet50)
- 🧠 Visual similarity matching with cosine similarity
- 👕➡️👖➡️👟 Intelligent outfit completion logic
- 🎛️ User preference filters (Color, Season, Usage)
- ⚡ Fast inference using precomputed embeddings & caching
- 🖥️ Interactive web interface built with Streamlit
- 🏷️ Brand-agnostic, style-focused recommendations

---

## 🧠 System Architecture
User Image
↓
CNN Feature Extraction (ResNet50)
↓
Cosine Similarity with Precomputed Embeddings
↓
Outfit Completion Logic
↓
Filtered & Ranked Recommendations
↓
Streamlit UI Output

## 📂 Project Structure
AI_Fashion_Project/
│
├── data/
│ ├── images/
│ └── styles.csv
│
├── embeddings/
│ ├── image_embeddings.npy
│ └── image_ids.npy
│
├── src/
│ ├── init.py
│ ├── config.py
│ ├── data_loader.py
│ ├── feature_extractor.py
│ ├── build_embeddings.py
│ ├── recommender.py
│ └── utils.py
│
├── app.py
├── requirements.txt
└── README.md

## 📊 Dataset
- **Source:** Kaggle – Fashion Product Images Dataset
- **Size:** ~44,000 images
- **Metadata:** Article type, color, season, usage

Brand information is intentionally excluded to keep the system **style-centric and brand-agnostic**.

---

## 🤖 Machine Learning Approach

### Feature Extraction
- Pretrained **ResNet50** model (ImageNet weights)
- Each image converted into a **2048-dimensional feature vector**

### Similarity Matching
- Cosine similarity used to find visually similar items
- Works even for external images not present in the dataset

### Outfit Completion Logic
- Upload Topwear → Recommend Bottomwear + Footwear
- Upload Bottomwear → Recommend Topwear + Footwear
- Upload Footwear → Recommend Topwear + Bottomwear

This combines **computer vision + fashion-aware rules**.

---

## ⚡ Performance Optimization
- CNN embeddings are **generated offline once**
- Runtime performs only similarity comparison
- Dataset and resources are cached
- Average response time **< 1 second**

---

## 🖥️ User Interface
- Built with **Streamlit**
- Features:
  - Image upload
  - Sidebar filters
  - Real-time recommendations
  - Clean and minimal design

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Generate Embeddings (One-Time)
bash
Copy code
python -m src.build_embeddings
This step may take several hours on CPU and is required only once.

3️⃣ Run the Application
bash
Copy code
streamlit run app.py