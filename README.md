# 🎬 DupVector Engine

This project is a multimodal video deduplication system that identifies **duplicate, variant, and distinct** video files using deep learning-based audio, video, and text similarity. Built with Streamlit for interactive dashboards, FAISS for vector search, and Transformer models for text embeddings.

---

## 🧠 Features

- 🔉 **Audio Fingerprinting** using MFCCs + cosine similarity
- 🎥 **Video Hashing** with perceptual image hashing (`pHash`)
- 📝 **Text Detection** from overlays using OCR + Transformer embeddings
- 🤖 **Similarity Search** with FAISS for fast vector comparisons
- 📊 **Interactive Dashboard** to visualize clusters, categories, and top matches
- 🎧 **Energy Plots** for audio comparison between files
- 🈶 **Language Detection** using Whisper for speech transcription
- 📂 **Duplicate Grouping** via Union-Find on pairwise similarity scores
- 🧮 **Clustering** via DBSCAN over weighted multimodal embeddings

---
## 🚀 Installation

```bash
git clone https://github.com/nsachdeva91/DupVector-Engine.git
cd DupVector-Engine
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🗃️ Requirements

Ensure the following are installed:

Python ≥ 3.10

FFmpeg (for Whisper transcription and audio loading)

Tesseract-OCR (for embedded text extraction)

MongoDB (optional for metadata persistence)

---

## 🧪 Run the Deduplication Dashboard


```bash
streamlit run dashboard.py
```
Upload a video file, or browse clustered duplicates from your indexed set.

---

## 📁 Directory Structure

<img width="355" height="104" alt="Image" src="https://github.com/user-attachments/assets/5c80fdfe-5b43-479d-ad39-683009868085" />

