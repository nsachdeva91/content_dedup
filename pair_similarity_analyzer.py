# video_similarity_analyzer.py
import os
import faiss
import numpy as np
import pickle
from collections import defaultdict, Counter
from itertools import combinations

class VideoSimilarityAnalyzer:
    def __init__(self, index_dir="faiss_indices", thresholds=None):
        self.index_dir = index_dir
        self.thresholds = thresholds or {
            "audio": 0.95,
            "video": 0.95,
            "text": 0.95
        }
        self._load_indices()
        self._load_metadata()
        self.results = []
        self.category_counter = Counter()
        self.graph = defaultdict(set)

    def _load_index(self, name):
        return faiss.read_index(os.path.join(self.index_dir, name))

    def _load_metadata(self):
        with open(os.path.join(self.index_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        self.filenames = [m["filename"] for m in self.metadata]

    def _load_indices(self):
        self.audio_index = self._load_index("audio.index")
        self.video_index = self._load_index("video.index")
        self.text_index = self._load_index("text.index")

        self.audio_vecs = np.array([self.audio_index.reconstruct(i) for i in range(self.audio_index.ntotal)])
        self.video_vecs = np.array([self.video_index.reconstruct(i) for i in range(self.video_index.ntotal)])
        self.text_vecs = np.array([self.text_index.reconstruct(i) for i in range(self.text_index.ntotal)])

    def cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) and np.linalg.norm(b) else 0.0

    def isDuplicate(self, a_sim, v_sim, t_sim):
        thresholds = {
            "audio": 0.99,
            "video": 0.99,
            "text": 0.99
        }
        return (a_sim >= thresholds["audio"] and
                v_sim >= thresholds["video"] and
                t_sim >= thresholds["text"])
    def get_match_label(self, a_sim, v_sim, t_sim):
        if self.isDuplicate(a_sim, v_sim, t_sim):
            return "Duplicate"
        a, v, t = (
            a_sim >= self.thresholds["audio"],
            v_sim >= self.thresholds["video"],
            t_sim >= self.thresholds["text"]
        )
        if a and v and t: return "Duplicate"
        elif a and v: return "Audio+Video"
        elif a and t: return "Audio+Text"
        elif v and t: return "Video+Text"
        elif a: return "Audio"
        elif v: return "Video"
        elif t: return "Text"
        return "No Match"

    def run_comparison(self, progress_callback=None):
        self.results.clear()
        self.category_counter.clear()
        self.graph.clear()

        total = len(self.filenames) * (len(self.filenames) - 1) / 2
        for idx, (i, j) in enumerate(combinations(range(len(self.filenames)), 2)):
            f1, f2 = self.filenames[i], self.filenames[j]
            a_sim = self.cosine_similarity(self.audio_vecs[i], self.audio_vecs[j])
            v_sim = self.cosine_similarity(self.video_vecs[i], self.video_vecs[j])
            t_sim = self.cosine_similarity(self.text_vecs[i], self.text_vecs[j])
            label = self.get_match_label(a_sim, v_sim, t_sim)

            self.category_counter[label] += 1
            self.results.append([f1, f2, f"{a_sim:.4f}", f"{v_sim:.4f}", f"{t_sim:.4f}", label])

            if label == "Duplicate":
                self.graph[f1].add(f2)
                self.graph[f2].add(f1)

            if progress_callback:
                progress_callback((idx + 1) / total, f"Comparing: {f1} vs {f2}")

    def get_results_table(self):
        return self.results

    def get_category_counts(self):
        return self.category_counter

    def get_duplicate_groups(self):
        visited = set()
        groups = []

        def dfs(node, group):
            if node in visited: return
            visited.add(node)
            group.append(node)
            for neighbor in self.graph[node]:
                dfs(neighbor, group)

        for node in self.graph:
            if node not in visited:
                group = []
                dfs(node, group)
                groups.append(sorted(group))

        return groups

    def get_distinct_files(self):
        duplicates_flat = set(v for g in self.get_duplicate_groups() for v in g)
        return sorted(set(self.filenames) - duplicates_flat)
