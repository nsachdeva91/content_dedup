import os
import faiss
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import pandas as pd
from pair_similarity_analyzer import VideoSimilarityAnalyzer
from itertools import combinations
import time
from collections import defaultdict, Counter
import networkx as nx
import tempfile


import subprocess
import certifi
import multiprocessing

META_PATH = "metadata.pkl"

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv

    def groups(self):
        from collections import defaultdict
        result = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            result[root].append(i)
        return list(result.values())

class VideoDeduplicator:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.audio = self._safe_load_faiss("audio.index", expected_dim=13)
        self.video = self._safe_load_faiss("video.index", expected_dim=64)
        self.text  = self._safe_load_faiss("text.index",  expected_dim=384)
        self.meta = self._load_meta()
        self.combined_index = None  # Will be created on demand

    def _safe_load_faiss(self, name, expected_dim):
        path = os.path.join(self.index_dir, name)
        index = faiss.read_index(path)
        vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
        mask = np.linalg.norm(vectors, axis=1) > 1e-3
        # Zero out any all-zero vectors
        return np.where(mask[:, None], vectors, np.zeros((vectors.shape[0], expected_dim), dtype=np.float32))

    def _load_meta(self):
        with open(os.path.join(self.index_dir, META_PATH), "rb") as f:
            return pickle.load(f)

    def get_combined_vectors(self, weights=(1.0, 1.0, 1.0)):
        audio_w, video_w, text_w = weights
        audio = normalize(self.audio) * audio_w
        video = normalize(self.video) * video_w
        text  = normalize(self.text)  * text_w

        max_len = max(len(audio), len(video), len(text))

        def pad(arr, target_len, vec_dim):
            if len(arr) == target_len:
                return arr
            padding = np.zeros((target_len - len(arr), vec_dim), dtype=arr.dtype)
            return np.vstack([arr, padding])

        audio = pad(audio, max_len, audio.shape[1])
        video = pad(video, max_len, video.shape[1])
        text  = pad(text,  max_len, text.shape[1])

        return np.concatenate([audio, video, text], axis=1)


    def cluster_videos(self, eps=0.3, min_samples=2, weights=(1.0, 1.0, 1.0)):
        combined = self.get_combined_vectors(weights)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(combined)
        return clustering.labels_

    def _build_combined_faiss_index(self, vectors):
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    
    def check_duplicate(self, vec, weights=(1.0, 1.0, 1.0)):
        all_vectors = self.get_combined_vectors(weights=weights)
        vec = normalize([vec])[0]
        base_vectors = all_vectors[:-1]

        if self.combined_index is None or self.combined_index.ntotal != base_vectors.shape[0]:
            self.combined_index = self._build_combined_faiss_index(base_vectors)

        vec = np.ascontiguousarray(vec.astype(np.float32))
        faiss.normalize_L2(vec.reshape(1, -1))
        scores, indices = self.combined_index.search(vec.reshape(1, -1), k=1)
        return indices[0][0], scores[0][0]

    def find_duplicate_pairs(self, threshold=0.90, weights=(1.0, 1.0, 1.0), top_k=5):
        results = []

        # Unpack weights
        w_audio, w_video, w_text = weights

        # Normalize individual modality vectors
        norm_audio = normalize(self.audio)
        norm_video = normalize(self.video)
        norm_text  = normalize(self.text)

        for i in range(len(self.meta)):
            silent_i = self.meta[i].get("silent_audio", False)
            for j in range(i + 1, len(self.meta)):
                silent_j = self.meta[j].get("silent_audio", False)

                if silent_i or silent_j:
                    # Only use video + text
                    v_sim = np.dot(norm_video[i], norm_video[j])
                    t_sim = np.dot(norm_text[i], norm_text[j])
                    combined_sim = (w_video * v_sim + w_text * t_sim) / (w_video + w_text)
                else:
                    # Use audio + video + text
                    a_sim = np.dot(norm_audio[i], norm_audio[j])
                    v_sim = np.dot(norm_video[i], norm_video[j])
                    t_sim = np.dot(norm_text[i], norm_text[j])
                    combined_sim = (
                        w_audio * a_sim + w_video * v_sim + w_text * t_sim
                    ) / (w_audio + w_video + w_text)

                if combined_sim >= threshold - 1e-6:
                    results.append((i, j, combined_sim))

        return results

    def get_duplicate_groups(self, threshold=0.90, weights=(1.0, 1.0, 1.0), top_k=5):
        pairs = self.find_duplicate_pairs(threshold=threshold, weights=weights, top_k=top_k)
        uf = UnionFind(len(self.meta))
        for i, j, _ in pairs:
            uf.union(i, j)
        return uf.groups()

    def _cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) and np.linalg.norm(b) else 0.0


    def get_similarity_components(self, idx1, idx2=None, override_vector=None):
        if override_vector is not None:
            # Decompose combined override vector
            audio_len = self.audio.shape[1]
            video_len = self.video.shape[1]

            audio_vec = override_vector[:audio_len]
            video_vec = override_vector[audio_len:audio_len + video_len]
            text_vec  = override_vector[audio_len + video_len:]

            a1 = normalize([self.audio[idx1]])[0]
            v1 = normalize([self.video[idx1]])[0]
            t1 = normalize([self.text[idx1]])[0]

            a2 = normalize([audio_vec])[0]
            v2 = normalize([video_vec])[0]
            t2 = normalize([text_vec])[0]
        else:
            a1 = normalize([self.audio[idx1]])[0]
            v1 = normalize([self.video[idx1]])[0]
            t1 = normalize([self.text[idx1]])[0]

            a2 = normalize([self.audio[idx2]])[0]
            v2 = normalize([self.video[idx2]])[0]
            t2 = normalize([self.text[idx2]])[0]

        audio_sim = self._cosine_similarity(a1, a2)
        video_sim = self._cosine_similarity(v1, v2)
        text_sim  = self._cosine_similarity(t1, t2)

        return {
            "audio": round(float(audio_sim), 4),
            "video": round(float(video_sim), 4),
            "text":  round(float(text_sim), 4)
        }

    def cluster_and_build_indices(self, eps=0.25, min_samples=2, weights=(1.0, 1.0, 1.0)):
        """
        Cluster videos using DBSCAN and build FAISS indices for each cluster
        Returns: (cluster_labels, cluster_indices, cluster_centroids)
        """
        # Get combined vectors for clustering
        combined_vectors = self.get_combined_vectors(weights)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(combined_vectors)
        cluster_labels = clustering.labels_
        
        # Group videos by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
        
        # Build FAISS indices for each cluster
        cluster_indices = {}
        cluster_centroids = {}
        
        for cluster_id, video_indices in clusters.items():
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get vectors for this cluster
            cluster_vectors = combined_vectors[video_indices]
            
            # Build FAISS index for this cluster
            cluster_vectors = np.ascontiguousarray(cluster_vectors.astype(np.float32))
            faiss.normalize_L2(cluster_vectors)
            
            dim = cluster_vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(cluster_vectors)
            
            # Store the index and mapping
            cluster_indices[cluster_id] = {
                'index': index,
                'video_indices': video_indices,  # Maps cluster positions to global video indices
                'vectors': cluster_vectors
            }
            
            # Calculate cluster centroid
            centroid = np.mean(cluster_vectors, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize
            cluster_centroids[cluster_id] = centroid
        
        return cluster_labels, cluster_indices, cluster_centroids
    
    def find_nearest_cluster(self, query_vector, cluster_centroids, weights=(1.0, 1.0, 1.0)):
        """
        Find the nearest cluster for a query vector
        Returns: (nearest_cluster_id, similarity_score)
        """
        # Normalize query vector
        query_combined = np.concatenate([
            query_vector["audio"] * weights[0],
            query_vector["video"] * weights[1],
            query_vector["text"] * weights[2]
        ]).astype(np.float32)
        
        query_combined = query_combined / np.linalg.norm(query_combined)
        
        best_cluster = -1
        best_similarity = -1
        
        for cluster_id, centroid in cluster_centroids.items():
            similarity = np.dot(query_combined, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        
        return best_cluster, best_similarity
    
    def search_in_cluster(self, query_vector, cluster_info, weights=(1.0, 1.0, 1.0), k=None):
        """
        Search for similar videos within a specific cluster
        Returns: List of (video_index, similarity_score)
        """
        # Prepare query vector
        query_combined = np.concatenate([
            query_vector["audio"] * weights[0],
            query_vector["video"] * weights[1],
            query_vector["text"] * weights[2]
        ]).astype(np.float32)
        
        query_combined = query_combined / np.linalg.norm(query_combined)
        query_combined = np.ascontiguousarray(query_combined.astype(np.float32))
        faiss.normalize_L2(query_combined.reshape(1, -1))
        
        # Search in cluster's FAISS index
        index = cluster_info['index']
        video_indices = cluster_info['video_indices']
        
        search_k = k if k is not None else len(video_indices)
        search_k = min(search_k, index.ntotal)
        
        similarities, indices = index.search(query_combined.reshape(1, -1), search_k)
        
        # Convert cluster-local indices to global video indices
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(video_indices):  # Valid index
                global_video_idx = video_indices[idx]
                results.append((global_video_idx, float(sim)))
        
        return results