from batch_processor import process_all_videos
from pymongo import MongoClient
from itertools import combinations
import os, pickle
import numpy as np
import joblib

import joblib
import shutil
import scipy
from scipy.spatial.distance import cdist
from VideoDeduplicator import VideoDeduplicator
from sklearn.metrics.pairwise import cosine_distances

INDEX_DIR = "faiss_indices"
def clear_existing_indices(index_dir="faiss_indices"):
    files_to_delete = ["audio.index", "video.index", "text.index", "metadata.pkl"]
    for filename in files_to_delete:
        path = os.path.join(index_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️ Deleted: {path}")

def removeAllData():
    try:
        client = MongoClient()
        db = client["video_dedup"]  
        result = db.videos.delete_many({})

    except Exception as e:
        print(f"❌ Error connecting/deleting to MongoDB: {e}")
    
    clear_existing_indices()


if __name__ == "__main__":
    removeAllData()
    process_all_videos("videos", downscale=(256, 256))