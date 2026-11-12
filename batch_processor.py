# batch_processor.py
import os
from multiprocessing import Pool, cpu_count
from pymongo import MongoClient
from video_processor import VideoProcessor
import time

INDEX_DIR = "faiss_indices"

def _safe_process_video(video_path, output_dir, downscale):
    try:
        db = MongoClient()["video_dedup"]
        vp = VideoProcessor(db, video_path, output_dir=output_dir, downscale=downscale)
        vp.process()
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")


def process_all_videos(folder, downscale=(256, 256), num_workers=None):
    start_time = time.time()
    video_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".mp4", ".mkv", ".mov"))
    ]
    args = [(v, os.path.join(INDEX_DIR, f"tmp_worker_{i}"), downscale) for i, v in enumerate(video_files)]
    num_workers = num_workers or max(1, cpu_count() - 1)

    # print(f"🚀 Processing {len(video_files)} videos with {num_workers} workers...")
    # with Pool(processes=num_workers) as pool:
    #     pool.starmap(_safe_process_video, args)
    
    for video_path in video_files:
        _safe_process_video(video_path, INDEX_DIR, downscale)

    print("📦 All videos processed")
    end_time = time.time()
    duration = end_time - start_time

    print(f"Time taken to process {len(video_files)} videos: {duration:.2f} seconds")
    # merger = IndexMerger(index_dir=INDEX_DIR, cleanup=True)
    # merger.merge()
    # print("🎉 Batch processing complete.")

