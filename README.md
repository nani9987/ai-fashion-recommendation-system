# AI-Powered Fashion Recommendation System 👕👖👟

## 📌 Project Overview
This project is an **AI-powered, image-first fashion recommendation system** that suggests visually similar clothing items based on **computer vision and deep learning**. Unlike traditional e-commerce platforms that rely on text search or purchase history, this system focuses on **visual similarity using CNN embeddings**, enabling style-focused recommendations.

### Key Capability
- Upload a clothing image → Get visually similar items ranked by similarity score
- OR browse by metadata filters (color, season, usage)
- Uses **ResNet50 CNN** for feature extraction + **cosine similarity** for matching

---

## 🎯 Key Features
- 📷 **Image-based recommendations** using CNN embeddings (ResNet50 - 2048-D vectors)
- 🧠 **Visual similarity matching** with cosine similarity metrics
- 👕➡️👖➡️👟 **Metadata-aware filtering** (Color, Season, Usage)
- ⚡ **Fast inference** using precomputed embeddings & caching
- 🖥️ **Interactive web interface** built with Streamlit
- 🏷️ **Brand-agnostic, style-focused** recommendations

---

## 🧠 System Architecture

```
User Input (Image)
    ↓
[CNN Feature Extraction - ResNet50]
    ↓
    [2048-D Feature Vector]
    ↓
[Cosine Similarity Search]
    ↓
[Top-N Similar Items]
    ↓
[Metadata Filtering (Optional)]
    ↓
[Ranked Recommendations]
    ↓
Streamlit UI Output
```

### Data Flow
1. **Offline Phase (One-time):**
   - Load 44K fashion images
   - Extract ResNet50 features for each image
   - Save embeddings as `.npy` files
   
2. **Online Phase (Per request):**
   - User uploads image or selects filters
   - Extract features from uploaded image
   - Compute cosine similarity with all precomputed embeddings
   - Return top-N matches sorted by similarity
   - Render in Streamlit UI

---

## 📂 Project Structure

```
ai-fashion-recommendation-system/
│
├── data/
│   ├── styles.csv                 # Metadata (44K rows)
│   └── images/                    # Fashion product images (44K JPGs)
│
├── embeddings/                    # Generated after build_embeddings.py
│   ├── image_embeddings.npy       # (44K, 2048) feature matrix
│   └── image_ids.npy              # Image IDs for lookup
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration & paths
│   ├── data_loader.py             # CSV loading & filtering
│   ├── feature_extractor.py       # ResNet50 feature extraction
│   ├── build_embeddings.py        # Generate embeddings (offline)
│   ├── recommender.py             # Cosine similarity ranking
│   └── utils.py                   # Helper functions
│
├── app.py                         # Streamlit UI
├── requirements.txt               # Python dependencies
├── README.md
└── .gitignore
```

---

## 📊 Dataset
- **Source:** [Kaggle – Fashion Product Images Dataset](https://www.kaggle.com/datasets/agamemnons/kag-risk-factors-for-heart-disease)
- **Size:** ~44,000 images
- **Format:** JPG images + CSV metadata
- **Metadata Columns:** `id`, `articleType`, `baseColour`, `season`, `usage`
- **Note:** Brand information is intentionally excluded to keep the system **style-centric**

---

## 🤖 Machine Learning Approach

### Feature Extraction
- **Model:** ResNet50 (pretrained on ImageNet)
- **Input:** 224×224 RGB images
- **Output:** 2048-dimensional feature vector per image
- **Why ResNet50?** Proven for image classification, transfers well to visual similarity tasks

### Similarity Matching
- **Metric:** Cosine similarity (normalized dot product)
- **Range:** 0 to 1 (1 = identical, 0 = completely different)
- **Why Cosine?** Works well in high-dimensional spaces, computationally efficient

### Performance Optimization
- CNN embeddings are **generated offline once** (takes ~2-3 hours on CPU)
- Runtime performs only **similarity comparison** (O(n) dot products)
- Dataset and embeddings are **cached** in memory
- **Average response time: < 1 second per recommendation**

---

## 🖥️ User Interface
- Built with **Streamlit** (fast prototyping)
- **Two search modes:**
  1. **Image Upload:** Upload a clothing item → Get visually similar recommendations
  2. **Filter Browsing:** Select color/season/usage → Browse matching items
- **Real-time recommendations** with similarity scores
- Clean, minimal design focused on user experience

---

## ▶️ How to Run the Project

### Prerequisites
- Python 3.8+
- 4GB+ RAM
- ~10GB disk space (for dataset + embeddings)

### 1️⃣ Clone and Setup
```bash
git clone https://github.com/nani9987/ai-fashion-recommendation-system.git
cd ai-fashion-recommendation-system

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset
Download the Kaggle Fashion Product Images dataset and extract to:
```
data/
├── styles.csv
└── images/
    ├── 1.jpg
    ├── 2.jpg
    └── ... (44K images)
```

[Kaggle Dataset Link](https://www.kaggle.com/datasets/agamemnons/kag-risk-factors-for-heart-disease)

### 3️⃣ Generate Embeddings (One-Time Setup)
```bash
python -m src.build_embeddings
```
**⏱️ Expected time:** 2-3 hours on CPU (30 mins on GPU)

**Output:**
```
embeddings/
├── image_embeddings.npy  # (44000, 2048) float32 array
└── image_ids.npy         # (44000,) int64 array
```

### 4️⃣ Run the Web App
```bash
streamlit run app.py
```

**Output:** Opens at `http://localhost:8501`

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Total Images** | 44,000 |
| **Feature Dimension** | 2,048 (ResNet50) |
| **Similarity Metric** | Cosine Similarity |
| **Avg Response Time** | ~500ms-1s |
| **Embedding Size** | ~350 MB (.npy format) |
| **Model** | ResNet50 (pretrained ImageNet) |

---

## 🔧 Troubleshooting

### Issue: "Embeddings not found"
**Solution:** Run `python -m src.build_embeddings` to generate embeddings

### Issue: "CSV not found"
**Solution:** Ensure `data/styles.csv` exists in the project root

### Issue: "No images found"
**Solution:** Verify images are in `data/images/` with names matching CSV IDs (e.g., `1.jpg`, `2.jpg`)

### Issue: Slow embedding generation
**Solution:** Use GPU acceleration (install `tensorflow[and-cuda]`) or reduce dataset size for testing

---

## 🚀 Future Improvements

- [ ] Add user feedback loop to retrain embeddings
- [ ] Implement approximate nearest neighbor search (FAISS) for faster retrieval
- [ ] Add outfit completion logic (e.g., "topwear → suggest bottomwear + footwear")
- [ ] Support multiple similarity metrics (Euclidean, Mahalanobis)
- [ ] Deploy to production (AWS/GCP/Azure)
- [ ] Add filtering by price, brand, rating
- [ ] Mobile app support

---

## 📝 License
MIT License - See LICENSE file

---

## 👨‍💻 Author
**nani9987**  
GitHub: [@nani9987](https://github.com/nani9987)

---

## 🙏 Acknowledgments
- ResNet50 architecture from Keras Applications
- Kaggle Fashion Product Images Dataset
- Streamlit for web UI framework
