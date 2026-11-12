import streamlit as st
import os
import faiss
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
from pair_similarity_analyzer import VideoSimilarityAnalyzer
from itertools import combinations
import time
from collections import defaultdict, Counter
import networkx as nx
from pyvis.network import Network
import tempfile

import librosa
import whisper
import matplotlib.pyplot as plt
import subprocess
import shutil
import certifi
import multiprocessing
from VideoDeduplicator import VideoDeduplicator, UnionFind
from video_processor import VideoProcessor
from pymongo import MongoClient
import cv2
from PIL import Image
import imagehash
import base64
import io

# Set environment to use trusted CA bundle
os.environ['SSL_CERT_FILE'] = certifi.where()


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


INDEX_DIR = "faiss_indices"
META_PATH = "metadata.pkl"
VIDEO_DIR = r"videos"


category_colors = {
    "Duplicate": "#ffcccc",        # light red
    "Audio+Video": "#ccffcc",      # light green
    "Video+Text": "#cce5ff",       # light blue
    "Audio+Text": "#ffe5cc",       # light orange
    "Video": "#e6ccff",            # light purple
    "Audio": "#ffffcc",            # light yellow
    "Text": "#d9f2d9",             # light mint green
    "No Match": "#f2f2f2",         # light gray
}


def render_clustered_duplicate_groups_with_filters(dedup, eps=0.1, weights=(1.0, 1.0, 1.0)):
    start_time = time.time()

    st.header("🔍 Clustered Videos + Pairwise Similarity")

    labels = dedup.cluster_videos(eps=eps, weights=weights)
    clustered = {}
    for label, meta in zip(labels, dedup.meta):
        clustered.setdefault(label, []).append(meta["filename"])

    analyzer = VideoSimilarityAnalyzer()
    duplicate_graph = defaultdict(set)
    distinct_videos = set()
    results_per_cluster = {}

    allowed_categories = {"Duplicate", "Audio+Video", "Video+Text", "Audio+Text"}

    pairwise_labels_per_file = defaultdict(list)
    pairwise_scores_per_file = defaultdict(list)
    variant_label_priority = [
        "Audio+Video", "Video+Text", "Audio+Text",
        "Audio Only", "Video Only", "Text Only"
    ]
    variant_video_reason = {}

    for cluster_id, files in clustered.items():
        if cluster_id == -1:
            distinct_videos.update(files)
            continue

        pairs = list(combinations(files, 2))
        rows = []

        for f1, f2 in pairs:
            i = analyzer.filenames.index(f1)
            j = analyzer.filenames.index(f2)
            a_sim = analyzer.cosine_similarity(analyzer.audio_vecs[i], analyzer.audio_vecs[j])
            v_sim = analyzer.cosine_similarity(analyzer.video_vecs[i], analyzer.video_vecs[j])
            t_sim = analyzer.cosine_similarity(analyzer.text_vecs[i], analyzer.text_vecs[j])
            label = analyzer.get_match_label(a_sim, v_sim, t_sim)

            pairwise_labels_per_file[f1].append(label)
            pairwise_labels_per_file[f2].append(label)

            pairwise_scores_per_file[f1].append((f2, a_sim, v_sim, t_sim, label))
            pairwise_scores_per_file[f2].append((f1, a_sim, v_sim, t_sim, label))

            if label in allowed_categories:
                rows.append([
                    f1,
                    f2,
                    f"{a_sim * 100:.2f}%",
                    f"{v_sim * 100:.2f}%",
                    f"{t_sim * 100:.2f}%",
                    label
                ])

            if label == "Duplicate":
                duplicate_graph[f1].add(f2)
                duplicate_graph[f2].add(f1)

        results_per_cluster[cluster_id] = (rows, files)

    # Show clusters with filtered rows
    for cluster_id, (rows, files) in results_per_cluster.items():
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])

        with st.expander(f"✅ Cluster {cluster_id + 1} ({len(df)} comparisons)", expanded=True):
            st.markdown("**Files in this cluster:**")
            for f in sorted(files):
                st.markdown(f"- {f}")

            def highlight_row_by_category(row):
                color = category_colors.get(row["Match Category"], "")
                return [f'background-color: {color}; color: black' if color else '' for _ in row]

            styled_df = df.style.apply(highlight_row_by_category, axis=1)
            st.dataframe(styled_df, use_container_width=True)

    # Use Union-Find to group duplicates
    filename_to_index = {f: i for i, f in enumerate(analyzer.filenames)}
    uf = UnionFind(len(analyzer.filenames))
    for f1, neighbors in duplicate_graph.items():
        for f2 in neighbors:
            uf.union(filename_to_index[f1], filename_to_index[f2])

    groups = []
    for group_indices in uf.groups():
        group = [analyzer.filenames[i] for i in group_indices]
        if len(group) > 1:
            groups.append(sorted(group))

    st.subheader("📂 Duplicate Groups")
    for i, group in enumerate(groups, 1):
        with st.expander(f"Group {i} ({len(group)} videos)", expanded=True):
            for f in sorted(group):
                st.markdown(f"- {f}")

    all_duplicates = set(v for g in groups for v in g)
    variant_videos = {}
    distinct_videos = set()

    for cluster_id, (rows, files) in results_per_cluster.items():
        if cluster_id == -1:
            continue

        for f in files:
            if f in all_duplicates:
                continue
            labels = pairwise_labels_per_file.get(f, [])
            if labels and all(l == "No Match" for l in labels):
                distinct_videos.add(f)
            else:
                # Assign best label and scores for tooltip
                for label in variant_label_priority:
                    if label in labels:
                        for (other_file, a_sim, v_sim, t_sim, l) in pairwise_scores_per_file[f]:
                            if l == label:
                                variant_videos[f] = {
                                    "reason": label,
                                    "score": f"Audio={a_sim*100:.2f}%, Video={v_sim*100:.2f}%, Text={t_sim*100:.2f}%",
                                    "color": category_colors.get(label, "")
                                }
                                break
                        break

    unclustered_videos = set()
    for label, files in clustered.items():
        if label == -1:
            unclustered_videos.update(files)

    # Show Variant Videos
    st.subheader("📁 Variant Videos")
    if variant_videos:
        with st.expander(f"{len(variant_videos)} Variant Files", expanded=True):
            for f, info in sorted(variant_videos.items()):
                reason = info["reason"]
                score = info["score"]
                color = info["color"]
                st.markdown(
                    f"<div style='background-color:{color}; color:black; padding:5px; border-radius:4px;'>"
                    f"🔹 <b>{f}</b><br/><small>Matched on: <i>{reason}</i> — {score}</small></div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No variant videos found.")

    # Show Distinct Videos (clustered but no matches)
    st.subheader("📂 Distinct Videos (in clusters with no matches)")
    if distinct_videos:
        with st.expander(f"{len(distinct_videos)} Distinct Files", expanded=True):
            for f in sorted(distinct_videos):
                st.markdown(
                    f"<div style='background-color:#ddd; color:black; padding:5px; border-radius:4px;'>"
                    f"🟦 <b>{f}</b> — No matching features in cluster</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No distinct videos found in clusters.")

    # Show Unclustered Distinct Videos
    st.subheader("📂 Unclustered Distinct Videos")
    if unclustered_videos:
        with st.expander(f"{len(unclustered_videos)} Unclustered Files", expanded=True):
            for f in sorted(unclustered_videos):
                st.markdown(
                    f"<div style='background-color:#f5f5f5; color:black; padding:5px; border-radius:4px;'>"
                    f"📄 <b>{f}</b> — Not clustered with any group</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No unclustered distinct videos found.")

    st.markdown(f"⏱ **Total processing time:** `{time.time() - start_time:.2f} seconds`")

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results

def detect_language_fast(text):
    try:
        cleaned = text.strip()
        return detect(cleaned) if len(cleaned) > 20 else "unknown"
    except:
        return "unknown"

        
CACHE_DIR = "cached_metadata"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_result(key, data):
    with open(os.path.join(CACHE_DIR, f"{key}.pkl"), "wb") as f:
        pickle.dump(data, f)

def load_cached(key):
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
    
def get_cached_audio_energy(filepath):
    key = f"rms_{os.path.basename(filepath)}"
    cached = load_cached(key)
    if cached: return cached
    result = compute_audio_energy(filepath)
    cache_result(key, result)
    return result

def get_cached_transcription(filepath):
    # return "", ""
    key = f"transcript_{os.path.basename(filepath)}"
    cached = load_cached(key)
    model = load_whisper_model()

    if cached:
        return cached["text"], cached["language"]  
    result = model.transcribe(filepath)
    cache_result(key, result)
    return result["text"], result["language"]

def extract_audio_signal(file_path, sample_rate=16000):
    """
    Extract mono float32 audio samples through FFmpeg.
    This avoids librosa/audioread MP4 decoder compatibility issues.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio/video file not found: {file_path}")

    command = [
        ffmpeg, "-v", "error", "-i", file_path,
        "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or "FFmpeg could not extract the audio stream.")

    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("The audio stream is empty.")

    return samples, sample_rate


def compute_audio_energy(file_path, silence_threshold=0.01,
                         min_silence_duration=1.0):
    """
    Return RMS timestamps, RMS values and detected silent intervals.
    """
    try:
        samples, sr = extract_audio_signal(file_path)
        frame_length = 1024
        hop_length = 512

        if samples.size < frame_length:
            samples = np.pad(samples, (0, frame_length - samples.size))

        rms_values = []
        rms_times = []

        for start in range(0, samples.size - frame_length + 1, hop_length):
            frame = samples[start:start + frame_length]
            rms_values.append(float(np.sqrt(np.mean(frame ** 2))))
            rms_times.append((start + frame_length / 2) / sr)

        rms = np.asarray(rms_values, dtype=np.float32)
        times = np.asarray(rms_times, dtype=np.float32)

        silent_segments = []
        silence_start = None
        frame_step = hop_length / sr

        for t, value in zip(times, rms):
            if value < silence_threshold:
                if silence_start is None:
                    silence_start = max(0.0, float(t) - frame_length / (2 * sr))
            elif silence_start is not None:
                silence_end = float(t)
                if silence_end - silence_start >= min_silence_duration:
                    silent_segments.append((silence_start, silence_end))
                silence_start = None

        if silence_start is not None:
            silence_end = samples.size / sr
            if silence_end - silence_start >= min_silence_duration:
                silent_segments.append((silence_start, silence_end))

        return times, rms, silent_segments

    except Exception as e:
        st.warning(
            f"⚠️ Could not load audio from "
            f"`{os.path.basename(file_path)}`: {e}"
        )
        return np.array([]), np.array([]), []


def plot_energy_comparison(file1, file2):
    """
    Plot waveform and RMS audio activity for two videos.
    """
    try:
        samples1, sr1 = extract_audio_signal(file1)
        samples2, sr2 = extract_audio_signal(file2)

        def make_rms(samples, sr, frame_length=1024, hop_length=512):
            if samples.size < frame_length:
                samples = np.pad(samples, (0, frame_length - samples.size))
            values, times = [], []
            for start in range(0, samples.size - frame_length + 1, hop_length):
                frame = samples[start:start + frame_length]
                values.append(float(np.sqrt(np.mean(frame ** 2))))
                times.append((start + frame_length / 2) / sr)
            return np.asarray(times), np.asarray(values)

        rms_times1, rms1 = make_rms(samples1, sr1)
        rms_times2, rms2 = make_rms(samples2, sr2)

        if rms1.size == 0 or rms2.size == 0:
            st.info("📭 Could not extract audio energy from one or both files.")
            return None

        duration1 = samples1.size / sr1
        duration2 = samples2.size / sr2

        # Limit waveform points for responsive rendering on long videos.
        def waveform_points(samples, sr, max_points=30000):
            step = max(1, int(np.ceil(samples.size / max_points)))
            x = np.arange(0, samples.size, step) / sr
            y = samples[::step]
            return x, y

        wave_x1, wave_y1 = waveform_points(samples1, sr1)
        wave_x2, wave_y2 = waveform_points(samples2, sr2)

        fig, (wave_ax, rms_ax) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=False,
            gridspec_kw={"height_ratios": [1, 1]}
        )

        wave_ax.plot(
            wave_x1, wave_y1,
            label=os.path.basename(file1), alpha=0.75, linewidth=0.6
        )
        wave_ax.plot(
            wave_x2, wave_y2,
            label=os.path.basename(file2), alpha=0.75, linewidth=0.6
        )
        wave_ax.set_title("Audio Waveform")
        wave_ax.set_ylabel("Amplitude")
        wave_ax.set_ylim(-1.05, 1.05)
        wave_ax.grid(True, alpha=0.25)
        wave_ax.legend(loc="upper right")

        rms_ax.plot(
            rms_times1, rms1,
            label=f"{os.path.basename(file1)} — RMS", alpha=0.85
        )
        rms_ax.plot(
            rms_times2, rms2,
            label=f"{os.path.basename(file2)} — RMS", alpha=0.85
        )
        rms_ax.axhline(
            0.01, linestyle="--", linewidth=1,
            label="Silence threshold (0.01)"
        )
        rms_ax.set_title("Audio Activity (RMS Energy)")
        rms_ax.set_xlabel("Time (seconds)")
        rms_ax.set_ylabel("RMS energy")
        rms_ax.grid(True, alpha=0.25)
        rms_ax.legend(loc="upper right")

        rms_ax.set_xlim(0, max(duration1, duration2))
        fig.suptitle("Audio Comparison: Uploaded Video vs Matched Video")
        fig.tight_layout()
        return fig

    except Exception as e:
        st.warning(f"⚠️ Could not create audio graphs: {e}")
        return None

def has_audio_stream(filepath):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip() != b''   

def get_scene_hashes_from_db(video_path):
    """Retrieve scene hashes from MongoDB for a given video path"""
    try:
        client = MongoClient()
        db = client["video_dedup"]
        
        # Try exact match first
        result = db.videos.find_one({"video_path": video_path})
        if result:
            return result.get("scene_hashes", [])
        
        # Try matching by filename if exact path doesn't work
        filename = os.path.basename(video_path)
        result = db.videos.find_one({"video_path": {"$regex": filename + "$"}})
        if result:
            return result.get("scene_hashes", [])
            
        return []
    except Exception as e:
        st.error(f"Error retrieving scene hashes from database: {e}")
        return []

def extract_frame_at_timestamp(video_path, timestamp):
    """Extract frame from video at given timestamp"""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        st.error(f"Error extracting frame: {e}")
        return None

def compare_scene_hashes(scene_hashes1, scene_hashes2, similarity_threshold=0.8):
    """Compare scene hashes between two videos and find matching scenes"""
    matches = []
    total_comparisons = len(scene_hashes1) * len(scene_hashes2)
    comparison_count = 0
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, scene1 in enumerate(scene_hashes1):
        for j, scene2 in enumerate(scene_hashes2):
            comparison_count += 1
            if total_comparisons > 0:
                progress = comparison_count / total_comparisons
                progress_bar.progress(progress)
                status_text.text(f"Comparing scenes: {i+1}/{len(scene_hashes1)} vs {j+1}/{len(scene_hashes2)}")
            
            # Compare hashes within each scene
            for hash1 in scene1.get("hashes", []):
                for hash2 in scene2.get("hashes", []):
                    # Calculate Hamming distance between hashes
                    try:
                        hamming_distance = bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
                        similarity = 1 - (hamming_distance / 64)  # Normalize for 64-bit hash
                        
                        if similarity > similarity_threshold:
                            matches.append({
                                "scene1_idx": i,
                                "scene2_idx": j,
                                "scene1_start": scene1["start"],
                                "scene1_end": scene1["end"],
                                "scene2_start": scene2["start"],
                                "scene2_end": scene2["end"],
                                "scene1_time": (scene1["start"] + scene1["end"]) / 2,
                                "scene2_time": (scene2["start"] + scene2["end"]) / 2,
                                "similarity": similarity,
                                "hash1": hash1,
                                "hash2": hash2
                            })
                    except ValueError:
                        continue
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Sort matches by similarity (descending)
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches

def extract_scene_clip(video_path, start_time, end_time, output_path):
    """Extract a video clip from start_time to end_time using ffmpeg"""
    try:
        # Ensure we have a reasonable clip length (at least 1 second, max 30 seconds)
        duration = max(1.0, min(30.0, end_time - start_time))
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-y', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            print(f"FFmpeg error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error extracting scene clip: {e}")
        return None

def display_frame_comparison(video1_path, video2_path, matches):
    """Display video scene comparison for matched scenes"""
    st.subheader("🎬 Scene Comparison Results")
    
    if not matches:
        st.info("No similar scenes found between the videos.")
        return
    
    # Show top matches
    max_matches = len(matches)#min(5, len(matches))
    st.write(f"Showing top {max_matches} similar scene pairs:")
    
    # Create a summary of similarity distribution
    if len(matches) > 0:
        similarities = [match['similarity'] for match in matches]
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        min_similarity = min(similarities)
        
        st.markdown("**📈 Similarity Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Similarity", f"{avg_similarity:.3f}")
        with col2:
            st.metric("Max Similarity", f"{max_similarity:.3f}")
        with col3:
            st.metric("Min Similarity", f"{min_similarity:.3f}")
    
    for i, match in enumerate(matches[:max_matches]):
        # Color code based on similarity
        if match['similarity'] >= 0.95:
            color = "green"
            similarity_label = "Very High"
        elif match['similarity'] >= 0.90:
            color = "blue"
            similarity_label = "High"
        elif match['similarity'] >= 0.85:
            color = "orange"
            similarity_label = "Medium"
        else:
            color = "red"
            similarity_label = "Low"
            
        with st.expander(f"Match {i+1} - Similarity: {match['similarity']:.3f} ({similarity_label})", expanded=i==0):
            st.markdown(f"**🔍 Scene Information:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Uploaded Video** (Scene {match['scene1_idx']+1})")
                st.write(f"⏱️ Scene Duration: {match['scene1_start']:.1f}s - {match['scene1_end']:.1f}s")
                st.write(f"⏱️ Scene Length: {match['scene1_end'] - match['scene1_start']:.1f}s")
                
                # Create temporary clip for uploaded video
                temp_clip1 = f"temp_clip1_{i}.mp4"
                with st.spinner("Extracting video clip..."):
                    clip1_path = extract_scene_clip(video1_path, match['scene1_start'], match['scene1_end'], temp_clip1)
                
                if clip1_path and os.path.exists(clip1_path):
                    st.video(clip1_path)
                    st.success("Video clip loaded successfully!")
                    # Clean up temporary file
                    try:
                        os.remove(clip1_path)
                    except:
                        pass
                else:
                    st.error("Could not extract scene clip from uploaded video")
                    # Fallback to frame extraction
                    frame1 = extract_frame_at_timestamp(video1_path, match['scene1_time'])
                    if frame1 is not None:
                        st.image(frame1, caption="Uploaded Video Frame (fallback)", use_column_width=True)
            
            with col2:
                st.markdown(f"**Database Video** (Scene {match['scene2_idx']+1})")
                st.write(f"⏱️ Scene Duration: {match['scene2_start']:.1f}s - {match['scene2_end']:.1f}s")
                st.write(f"⏱️ Scene Length: {match['scene2_end'] - match['scene2_start']:.1f}s")
                
                # Create temporary clip for database video
                temp_clip2 = f"temp_clip2_{i}.mp4"
                with st.spinner("Extracting video clip..."):
                    clip2_path = extract_scene_clip(video2_path, match['scene2_start'], match['scene2_end'], temp_clip2)
                
                if clip2_path and os.path.exists(clip2_path):
                    st.video(clip2_path)
                    st.success("Video clip loaded successfully!")
                    # Clean up temporary file
                    try:
                        os.remove(clip2_path)
                    except:
                        pass
                else:
                    st.error("Could not extract scene clip from database video")
                    # Fallback to frame extraction
                    frame2 = extract_frame_at_timestamp(video2_path, match['scene2_time'])
                    if frame2 is not None:
                        st.image(frame2, caption="Database Video Frame (fallback)", use_column_width=True)
            
            st.markdown(f"**🔍 Hash Comparison:**")
            st.code(f"Hash 1: {match['hash1']}")
            st.code(f"Hash 2: {match['hash2']}")
            st.markdown(f"**Similarity Score:** <span style='color:{color}'>{match['similarity']:.3f}</span>", unsafe_allow_html=True)

def display_video_comparison_expander(uploaded_video_path, result_row, category_title):
    """Display a single video comparison in an expander format"""
    rank, filename, audio_sim, video_sim, text_sim, cluster_sim, label = result_row
    
    # Get the color for this category
    color = category_colors.get(label, "#f2f2f2")
    
    # Create expander title with match details
    expander_title = f"🎬 {filename} - {label} (A:{audio_sim} V:{video_sim} T:{text_sim})"
    
    with st.expander(expander_title, expanded=False):
        st.markdown(f"**🔍 Match Details:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Audio Similarity", audio_sim)
        with col2:
            st.metric("Video Similarity", video_sim)
        with col3:
            st.metric("Text Similarity", text_sim)
        with col4:
            st.metric("Cluster Similarity", cluster_sim)
        
        # Show match category with color
        st.markdown(f"**Match Category:** <span style='background-color:{color}; color:black; padding:3px 8px; border-radius:4px;'>{label}</span>", unsafe_allow_html=True)
        
        # Display videos side by side
        st.markdown("**📹 Video Comparison:**")
        col1, col2 = st.columns(2)
        
        database_video_path = os.path.join(VIDEO_DIR, filename)
        
        with col1:
            st.markdown(f"**🎬 Uploaded Video**")
            if os.path.exists(uploaded_video_path):
                st.video(uploaded_video_path)
            else:
                st.error("Uploaded video not found")
                
        with col2:
            st.markdown(f"**🎥 Database Video** ({filename})")
            if os.path.exists(database_video_path):
                st.video(database_video_path)
            else:
                st.error("Database video not found")

def main():
    # === STREAMLIT DASHBOARD ===
    st.set_page_config(layout="wide")
    st.title("📼 Video Deduplication Dashboard")
    
    dedup = VideoDeduplicator(INDEX_DIR)

    option = st.sidebar.selectbox("Choose Action", ["New upload", "View Clusters"])

    if option == "View Clusters":
        eps = st.slider("Clustering Sensitivity (lower = stricter)", 0.05, 0.5, 0.25, 0.05)
        render_clustered_duplicate_groups_with_filters(dedup, eps=eps)

    elif option == "New upload":
        st.header("🧪 Check If a Video is a Duplicate (DBSCAN Clustering)")

        video_path = st.text_input("Enter full path to video file:", value="Sample_Video_720p_SpanishAudio.mp4")
        eps = st.slider("Clustering Sensitivity (eps)", 0.05, 0.5, 0.25, 0.05)
        min_samples = 2#st.slider("Minimum Samples per Cluster", 2, 10, 2, 1)
        
        start_time = time.time()
        if video_path and os.path.isfile(video_path):
            st.text("Processing selected video...")
            
            # Step 1: Process the uploaded video
            db = MongoClient()["video_dedup"]
            
            # Check if video is already in database, if not process it
            existing_video = db.videos.find_one({"video_path": video_path})
            if not existing_video:
                st.info("Video not found in database. Processing and storing scene hashes...")
                vp = VideoProcessor(db, video_path, output_dir=INDEX_DIR)
                vp.process()  # This will store scene hashes in MongoDB
            
            vp = VideoProcessor(db, video_path, output_dir=INDEX_DIR)
            vecs = vp.compute_vectors()

            is_silent = vecs["silent_audio"]
            audio_weight = 0.0 if is_silent else 1.0

            if is_silent:
                st.warning("⚠️ This video has no audio — only comparing text and video.")
            
            weights = (audio_weight, 1.0, 1.0)
            
            # Step 2: Cluster existing videos using DBSCAN
            st.text("Clustering existing videos...")
            cluster_labels, cluster_indices, cluster_centroids = dedup.cluster_and_build_indices(
                eps=eps, min_samples=min_samples, weights=weights
            )
            
            # Show clustering info
            n_clusters = len(cluster_centroids)
            n_noise = np.sum(cluster_labels == -1)
            st.info(f"Found {n_clusters} clusters")
            
            # Step 3: Find nearest cluster for the uploaded video
            st.text("Finding nearest cluster...")
            nearest_cluster, cluster_similarity = dedup.find_nearest_cluster(
                vecs, cluster_centroids, weights=weights
            )
            
            if nearest_cluster == -1:
                st.warning("No suitable cluster found. The video might be very different from existing videos.")
                st.stop()
            
            st.success(f"Nearest cluster: {nearest_cluster} (similarity: {cluster_similarity:.4f})")
            
            # Step 4: Search within the nearest cluster
            st.text("Searching within nearest cluster...")
            cluster_info = cluster_indices[nearest_cluster]
            cluster_results = dedup.search_in_cluster(vecs, cluster_info, weights=weights)
            
            # Step 5: Get detailed similarity components and labels
            analyzer = VideoSimilarityAnalyzer()
            results = []
            distinct = []
            moderate_variants = []  # Audio+Video, Video+Text, Audio+Text
            low_variants = []       # Video, Audio, Text only
            duplicates = []

            first_match_fig = None
            first_match_lang_block = ""
            first_match_shown = False

            for rank, (video_idx, overall_sim) in enumerate(cluster_results, start=1):
                meta = dedup.meta[video_idx]
                filename = meta["filename"]
                
                # Get combined vector for similarity calculation
                combined = np.concatenate([
                    vecs["audio"] * audio_weight,
                    vecs["video"],
                    vecs["text"]
                ]).astype(np.float32)
                
                similarities = dedup.get_similarity_components(video_idx, idx2=None, override_vector=combined)
                
                a_sim = similarities['audio']
                v_sim = similarities['video']
                t_sim = similarities['text']

                label = analyzer.get_match_label(a_sim, v_sim, t_sim)
                normalized_label = label.strip().lower()

                full_match_path = os.path.join(VIDEO_DIR, filename)

                # Build the audio graphs for the first result pair that has audio.
                # Do not restrict this to particular match categories.
                if not first_match_shown:
                    if os.path.isfile(video_path) and os.path.isfile(full_match_path):
                        try:
                            uploaded_has_audio = has_audio_stream(video_path)
                            matched_has_audio = has_audio_stream(full_match_path)

                            if uploaded_has_audio and matched_has_audio:
                                first_match_fig = plot_energy_comparison(
                                    video_path, full_match_path
                                )
                                first_match_shown = first_match_fig is not None

                                # Keep language analysis, but don't let transcription
                                # errors prevent the audio graphs from being displayed.
                                try:
                                    text_a = get_cached_transcription(video_path)[0]
                                    text_b = get_cached_transcription(full_match_path)[0]
                                    lang_a = detect_language_fast(text_a)
                                    lang_b = detect_language_fast(text_b)

                                    first_match_lang_block = (
                                        f"<div style='margin-top:20px'><b>🈶 Language Analysis:</b><br/>"
                                        f"Uploaded video: <code>{lang_a}</code><br/>"
                                        f"Matched video: <code>{lang_b}</code><br/>"
                                    )
                                    if lang_a != lang_b:
                                        first_match_lang_block += (
                                            "<span style='color:red'>"
                                            "⚠️ Language mismatch detected.</span>"
                                        )
                                    first_match_lang_block += "</div>"
                                except Exception as transcription_error:
                                    st.warning(
                                        f"Language analysis could not be completed: "
                                        f"{transcription_error}"
                                    )
                            # If this pair has no audio, leave first_match_shown
                            # false so a later result pair can be checked.
                        except Exception as audio_check_error:
                            st.warning(
                                f"Audio comparison could not be prepared: "
                                f"{audio_check_error}"
                            )
                    else:
                        st.warning("❗ One or both video files are missing.")

                result_row = [
                    rank,
                    filename,
                    f"{a_sim * 100:.2f}%",
                    f"{v_sim * 100:.2f}%",
                    f"{t_sim * 100:.2f}%",
                    f"{overall_sim * 100:.2f}%",
                    label
                ]
                results.append(result_row)

                # Categorize the videos based on similarity type
                if label == "Duplicate":
                    duplicates.append(result_row)
                elif label in {"No Match"}:
                    distinct.append(result_row)
                elif label in {"Audio+Video", "Video+Text", "Audio+Text"}:
                    moderate_variants.append(result_row)
                elif label in {"Video", "Audio", "Text"}:
                    low_variants.append(result_row)
                else:
                    # Catch any other labels that might exist
                    moderate_variants.append(result_row)

            # Show cluster information
            st.subheader(f"🔗 Results from Cluster {nearest_cluster}")
            st.write(f"Cluster contains {len(cluster_results)} videos")
            
            # Show main results table
            df = pd.DataFrame(results, columns=[
                "Rank", "Filename", "AudioSim", "VideoSim", "TextSim", "ClusterSim", "Match Category"
            ])

            # Optional: Color styling for the match categories
            def highlight_row_by_category(row):
                color = category_colors.get(row["Match Category"], "")
                return [f'background-color: {color}; color: black' if color else '' for _ in row]

            st.dataframe(df.style.apply(highlight_row_by_category, axis=1), use_container_width=True)

            # Display distinct, variants, and duplicates with similarity scores
            st.subheader("📊 Video Categorization")
            
            st.write("### 🔴 Duplicate Videos")
            if duplicates:
                st.write(f"Found {len(duplicates)} duplicate videos:")
                for result_row in duplicates:
                    display_video_comparison_expander(video_path, result_row, "Duplicate")
            else:
                st.info("No duplicate videos found in this cluster.")
            
            st.write("### 🟡 Moderate Similarity Videos (Audio+Video, Video+Text, Audio+Text)")
            if moderate_variants:
                st.write(f"Found {len(moderate_variants)} moderate similarity videos:")
                for result_row in moderate_variants:
                    display_video_comparison_expander(video_path, result_row, "Moderate Similarity")
            else:
                st.info("No moderate similarity videos found in this cluster.")
            
            st.write("### 🟠 Low Similarity Videos (Single feature matches: Video, Audio, or Text only)")
            if low_variants:
                st.write(f"Found {len(low_variants)} low similarity videos:")
                for result_row in low_variants:
                    display_video_comparison_expander(video_path, result_row, "Low Similarity")
            else:
                st.info("No low similarity videos found in this cluster.")
            
            st.write("### 🔵 Distinct Videos")
            if distinct:
                st.write(f"Found {len(distinct)} distinct videos:")
                for result_row in distinct:
                    display_video_comparison_expander(video_path, result_row, "Distinct")
            else:
                st.info("No distinct videos found in this cluster.")

            if first_match_fig:
                st.markdown("### 🎧 Audio Waveform & Activity Comparison")
                st.caption(
                    "Top graph: audio waveform amplitude over time. "
                    "Bottom graph: RMS energy; values below the dashed threshold "
                    "are treated as low-activity/silent audio."
                )
                st.pyplot(first_match_fig, use_container_width=True)
            else:
                st.info(
                    "Audio graphs are not available because no result pair with "
                    "readable audio streams was found."
                )
            if first_match_lang_block:
                st.markdown(first_match_lang_block, unsafe_allow_html=True)
                
            # Frame Comparison Section
            st.markdown("### Scene Comparison")
            st.markdown("Select a video from the results above to compare Scenes:")
            
            # Get filenames for selection
            result_filenames = [row[1] for row in results]
            
            if result_filenames:
                selected_video = st.selectbox(
                    "Select video for Scene comparison:",
                    ["None"] + result_filenames,
                    index=0
                )
                
                if selected_video != "None":
                    selected_video_path = os.path.join(VIDEO_DIR, selected_video)
                    
                    # Add similarity threshold control
                    similarity_threshold = st.slider(
                        "Scene Similarity Threshold",
                        0.7, 1.0, 0.8, 0.05,
                        help="Higher values = more strict matching"
                    )
                    
                    # if st.button("Compare Scenes"):
                    with st.spinner("Retrieving scene hashes..."):
                        # Get scene hashes for both videos
                        uploaded_scene_hashes = get_scene_hashes_from_db(video_path)
                        selected_scene_hashes = get_scene_hashes_from_db(selected_video_path)
                        
                        if uploaded_scene_hashes and selected_scene_hashes:
                            # Compare scene hashes
                            matches = compare_scene_hashes(
                                uploaded_scene_hashes,
                                selected_scene_hashes,
                                similarity_threshold
                            )
                            
                            # Display frame comparison
                            display_frame_comparison(video_path, selected_video_path, matches)
                            
                            # Show scene statistics
                            st.subheader("📊 Scene Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Uploaded Video Scenes", len(uploaded_scene_hashes))
                            with col2:
                                st.metric("Selected Video Scenes", len(selected_scene_hashes))
                            with col3:
                                st.metric("Similar Scene Pairs Found", len(matches))
                            
                            if matches:
                                st.success(f"Found {len(matches)} similar Scene pairs!")
                            else:
                                st.info("No similar Scenes found. Try lowering the similarity threshold.")
                            
                        else:
                            st.warning("Could not retrieve scene hashes from database. Make sure both videos are processed.")
                        
            else:
                st.info("No videos found in results to compare.")
        else:
            st.warning("Please enter a valid path")
        
        st.markdown(f"⏱ **Total processing time:** `{time.time() - start_time:.2f} seconds`")

    elif option == "Batch Deduplication Report":
        st.header("Similar Video Pairs")
        threshold = st.slider("Similarity threshold", 0.95, 1.0, 0.98, 0.01)
        max_pairs = st.number_input("Max pairs to display", 1, 1000, 100)
        
        with st.spinner("Scanning for similar pairs..."):
            pairs = dedup.find_duplicate_pairs(threshold=threshold)
        
        st.success(f"Found {len(pairs)} similar pairs")

        for i, (idx1, idx2, sim) in enumerate(sorted(pairs, key=lambda x: -x[2])[:max_pairs]):
            meta1 = dedup.meta[idx1]
            meta2 = dedup.meta[idx2]
            similarities = dedup.get_similarity_components(idx1, idx2)

            st.markdown(f"""
            🔁 **{meta1['filename']}** ↔ **{meta2['filename']}**  
            **Combined Similarity:** `{sim:.4f}`  
            - 🎧 Audio: `{similarities['audio']:.4f}`  
            - 🖼️ Video: `{similarities['video']:.4f}`  
            - 📝 Text: `{similarities['text']:.4f}`
            """)
            if dedup.meta[idx1]["silent_audio"] or dedup.meta[idx2]["silent_audio"]:
                st.warning("⚠️ One or both videos have no audio — audio similarity may be meaningless.")

        st.header("Duplicate Groups & Distinct Videos")
        groups = dedup.get_duplicate_groups(threshold=threshold)

        duplicates = [g for g in groups if len(g) > 1]
        singles = [g[0] for g in groups if len(g) == 1]

        st.subheader(f"✅ {len(duplicates)} Duplicate Groups")
        for i, group in enumerate(duplicates):
            st.markdown(f"**Group {i + 1}**:")
            for idx in group:
                st.text(f"📼 {dedup.meta[idx]['filename']}")
            st.markdown("---")

        st.subheader(f"🧩 {len(singles)} Distinct Videos")
        for idx in singles:
            st.text(f"📼 {dedup.meta[idx]['filename']}")


    elif option == "Pair similarity detailed":
        
        st.title("🎥 Video Similarity & Deduplication Dashboard")
        st.markdown("This compares all videos and groups duplicates using audio, video, and text similarity.")

        analyzer = VideoSimilarityAnalyzer()

        progress_bar = st.progress(0.0, text="Starting...")
        analyzer.run_comparison(progress_callback=lambda p, msg: progress_bar.progress(p, text=msg))
        progress_bar.empty()
        st.success("✅ Comparison complete.")

        # Table
        st.subheader("📋 Pairwise Comparison Results")
        df = pd.DataFrame(analyzer.get_results_table(), columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])
        st.dataframe(df, use_container_width=True)

        # Download CSV
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="video_similarity_results.csv")

        # Category Distribution
        st.subheader("📊 Match Category Distribution")
        counts = analyzer.get_category_counts()
        labels = list(counts.keys())
        values = [counts[l] for l in labels]
        colors = ["#4CAF50" if "Match" in l else "#F44336" for l in labels]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')
        st.pyplot(fig)

        # Duplicate Groups
        st.subheader("📂 Duplicate Groups")
        groups = analyzer.get_duplicate_groups()
        if groups:
            for i, group in enumerate(groups, 1):
                st.markdown(f"**Group {i}:** {', '.join(group)}")
        else:
            st.info("No duplicate groups found.")

        # Distinct Videos
        st.subheader("📁 Distinct Videos")
        distinct = analyzer.get_distinct_files()
        if distinct:
            for f in distinct:
                st.markdown(f"- {f}")
        else:
            st.info("All videos are part of duplicate groups.")

    elif option == "Pair similarity detailed":
        
        st.title("🎥 Video Similarity & Deduplication Dashboard")
        st.markdown("This compares all videos and groups duplicates using audio, video, and text similarity.")

        analyzer = VideoSimilarityAnalyzer()

        progress_bar = st.progress(0.0, text="Starting...")
        analyzer.run_comparison(progress_callback=lambda p, msg: progress_bar.progress(p, text=msg))
        progress_bar.empty()
        st.success("✅ Comparison complete.")

        # Table
        st.subheader("📋 Pairwise Comparison Results")
        df = pd.DataFrame(analyzer.get_results_table(), columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])
        st.dataframe(df, use_container_width=True)

        # Download CSV
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="video_similarity_results.csv")

        # Category Distribution
        st.subheader("📊 Match Category Distribution")
        counts = analyzer.get_category_counts()
        labels = list(counts.keys())
        values = [counts[l] for l in labels]
        colors = ["#4CAF50" if "Match" in l else "#F44336" for l in labels]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, color=colors)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')
        st.pyplot(fig)

        # Duplicate Groups
        st.subheader("📂 Duplicate Groups")
        groups = analyzer.get_duplicate_groups()
        if groups:
            for i, group in enumerate(groups, 1):
                st.markdown(f"**Group {i}:** {', '.join(group)}")
        else:
            st.info("No duplicate groups found.")

        # Distinct Videos
        st.subheader("📁 Distinct Videos")
        distinct = analyzer.get_distinct_files()
        if distinct:
            for f in distinct:
                st.markdown(f"- {f}")
        else:
            st.info("All videos are part of duplicate groups.")


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()