# video_processor.py

import os
import time
import cv2
import pickle
import shutil
import subprocess
import tempfile
import wave
import re

import numpy as np
import faiss
import imagehash
import pytesseract

from PIL import Image
from datetime import datetime
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector
from python_speech_features import mfcc
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


# === CONFIGURATION ===
SAMPLE_RATE = 16000
META_PATH = "metadata.pkl"
AUDIO_DIM = 13
VIDEO_DIM = 64
TEXT_DIM = 384

# Load once and reuse.
embedder = SentenceTransformer("all-MiniLM-L6-v2")


class VideoProcessor:
    def __init__(
        self,
        db,
        video_path,
        output_dir,
        downscale=(256, 256),
        skip_frames=2,
        threshold=30.0
    ):
        self.db = db
        self.video_path = os.path.abspath(video_path)
        self.downscale = downscale
        self.skip_frames = skip_frames
        self.threshold = threshold
        self.output_dir = os.path.abspath(output_dir)

    def _detect_scenes(self):
        video = open_video(self.video_path)
        video.skip_frames = self.skip_frames

        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=self.threshold))

        video.seek(0)
        manager.detect_scenes(video)
        scenes = manager.get_scene_list()

        if not scenes:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            duration = frame_count / fps if fps and fps > 0 else 0
            fps_for_timecode = video.frame_rate
            scenes = [
                (
                    FrameTimecode(0, fps=fps_for_timecode),
                    FrameTimecode(duration, fps=fps_for_timecode)
                )
            ]

        return [
            (scene[0].get_seconds(), scene[1].get_seconds())
            for scene in scenes
        ]

    def _compute_frame_at_time(self, cap, timestamp):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()

        if not ret or frame is None:
            return None, None, None

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.downscale:
            img = cv2.resize(
                img,
                self.downscale,
                interpolation=cv2.INTER_AREA
            )

        text = pytesseract.image_to_string(img)
        frame_hash = imagehash.phash(Image.fromarray(img))
        hash_vec = frame_hash.hash.flatten().astype(np.uint8)

        return frame_hash, hash_vec, text.strip()

    def _compute_frame(self, cap, start, end):
        mid_sec = (start + end) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000)

        ret, frame = cap.read()
        if not ret or frame is None:
            return None, None, None

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.downscale:
            img = cv2.resize(
                img,
                self.downscale,
                interpolation=cv2.INTER_AREA
            )

        text = pytesseract.image_to_string(img)
        frame_hash = imagehash.phash(Image.fromarray(img))
        hash_vec = frame_hash.hash.flatten().astype(np.uint8)

        return frame_hash, hash_vec, text.strip()

    def _load_faiss_index(self, index_path, dim):
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)

            if index.d != dim:
                raise ValueError(
                    f"FAISS dimension mismatch for {index_path}: "
                    f"expected {dim}, found {index.d}. "
                    "Back up and rebuild this index."
                )

            return index

        return faiss.IndexFlatIP(dim)

    def _save_faiss_index(self, index, index_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)

    @staticmethod
    def normalize_vec(vec):
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        return normalize(vec)[0].astype(np.float32)

    def _text_embedding(self, texts):
        text = " ".join([t for t in texts if t]).strip().lower()
        clean = re.sub(r"\s+", " ", text)

        if not clean:
            return np.zeros(TEXT_DIM, dtype=np.float32)

        return embedder.encode(clean).astype(np.float32)

    def _extract_audio(self):
        """
        Extract audio with FFmpeg, then calculate the existing 13 MFCC features.

        Returns:
            (mfcc_vector, total_duration_seconds, active_duration_seconds)

        If the video has no audio stream, returns:
            (None, 0.0, 0.0)

        If an audio stream exists but is silent, returns a zero MFCC vector,
        its duration, and 0.0 active seconds.
        """
        video_path = os.path.abspath(self.video_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")

        if not ffmpeg or not ffprobe:
            raise RuntimeError(
                "FFmpeg and FFprobe must be installed and available on PATH."
            )

        # Check whether the input contains an audio stream.
        probe = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                video_path
            ],
            capture_output=True,
            text=True
        )

        if probe.returncode != 0:
            raise RuntimeError(
                f"FFprobe failed for {video_path}: {probe.stderr.strip()}"
            )

        if not probe.stdout.strip():
            print(f"[AUDIO] No audio stream found: {video_path}")
            return None, 0.0, 0.0

        # Use a unique temporary file to avoid collisions between videos.
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="video_audio_",
            delete=False
        )
        wav_path = temp_file.name
        temp_file.close()

        try:
            extraction = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v", "error",
                    "-i", video_path,
                    "-map", "0:a:0",
                    "-vn",
                    "-ac", "1",
                    "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    wav_path
                ],
                capture_output=True,
                text=True
            )

            if extraction.returncode != 0:
                raise RuntimeError(
                    "FFmpeg audio extraction failed: "
                    f"{extraction.stderr.strip()}"
                )

            # Read PCM WAV data without depending on Pydub's MP4 decoder.
            with wave.open(wav_path, "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frame_count = wav.getnframes()
                raw_frames = wav.readframes(frame_count)

            if sample_rate != SAMPLE_RATE:
                raise ValueError(
                    f"Unexpected WAV sample rate: {sample_rate}"
                )

            if channels != 1:
                raise ValueError(
                    f"Expected mono WAV after FFmpeg conversion; got {channels} channels"
                )

            if sample_width != 2:
                raise ValueError(
                    f"Expected 16-bit PCM WAV; got {sample_width * 8}-bit"
                )

            samples = (
                np.frombuffer(raw_frames, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            if samples.size == 0:
                raise ValueError("Extracted audio is empty.")

            total_sec = float(samples.size / SAMPLE_RATE)

            # Measure active audio in 0.5-second windows.
            window_size = int(SAMPLE_RATE * 0.5)
            active_sec = 0.0

            for start in range(0, samples.size, window_size):
                window = samples[start:start + window_size]
                if window.size == 0:
                    continue

                # RMS is a more direct energy measure than standard deviation.
                rms = float(np.sqrt(np.mean(np.square(window))))
                if rms > 0.01:
                    active_sec += window.size / SAMPLE_RATE

            active_sec = min(active_sec, total_sec)

            # Treat an audio stream with no meaningful signal as silent.
            if active_sec <= 0:
                print(f"[AUDIO] Audio stream is silent: {video_path}")
                return (
                    np.zeros(AUDIO_DIM, dtype=np.float32),
                    total_sec,
                    0.0
                )

            mfcc_features = mfcc(
                samples,
                samplerate=SAMPLE_RATE,
                numcep=AUDIO_DIM
            )

            if mfcc_features.shape[0] == 0:
                raise ValueError("No MFCC features extracted.")

            mfcc_vec = np.mean(
                mfcc_features,
                axis=0
            ).astype(np.float32)

            if mfcc_vec.shape[0] != AUDIO_DIM:
                raise ValueError(
                    f"Expected {AUDIO_DIM} MFCC values, "
                    f"got {mfcc_vec.shape[0]}"
                )

            return mfcc_vec, total_sec, active_sec

        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError as cleanup_error:
                    print(
                        f"[WARNING] Could not remove temporary WAV "
                        f"{wav_path}: {cleanup_error}"
                    )

    def process(self):
        if self.db.videos.find_one({"video_path": self.video_path}):
            print(
                f"Skipping {self.video_path}: already present in MongoDB."
            )
            return

        start_time = time.time()
        os.makedirs(self.output_dir, exist_ok=True)

        audio_index_path = os.path.join(self.output_dir, "audio.index")
        video_index_path = os.path.join(self.output_dir, "video.index")
        text_index_path = os.path.join(self.output_dir, "text.index")

        audio_index = self._load_faiss_index(audio_index_path, AUDIO_DIM)
        video_index = self._load_faiss_index(video_index_path, VIDEO_DIM)
        text_index = self._load_faiss_index(text_index_path, TEXT_DIM)

        # === 1. Detect scenes and extract frame features ===
        scenes = self._detect_scenes()
        print(
            f"Scenes detected for {self.video_path}: {len(scenes)}"
        )

        scene_hashes = []
        texts = []
        hash_vectors = []

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"OpenCV could not open video: {self.video_path}"
            )

        try:
            for start, end in scenes:
                timestamps = [(start + end) / 2]
                hashes = []

                for timestamp in timestamps:
                    frame_hash, hash_vec, text = (
                        self._compute_frame_at_time(cap, timestamp)
                    )

                    if frame_hash is not None:
                        hashes.append(str(frame_hash))
                        hash_vectors.append(hash_vec)

                    if text:
                        texts.append(text)

                scene_hashes.append({
                    "start": start,
                    "end": end,
                    "hashes": hashes
                })
        finally:
            cap.release()

        # === 2. Aggregate and store video pHash vector ===
        if hash_vectors:
            video_vec = np.mean(
                hash_vectors,
                axis=0
            ).astype(np.float32)

            norm_video_vec = self.normalize_vec(video_vec)
            video_index.add(
                np.asarray([norm_video_vec], dtype=np.float32)
            )

        # === 3. Aggregate and store text embedding vector ===
        if texts:
            text_vec = self._text_embedding(texts)
        else:
            print(
                f"[WARNING] No text found in {self.video_path}; "
                "using a zero vector."
            )
            text_vec = np.zeros(TEXT_DIM, dtype=np.float32)

        norm_text_vec = self.normalize_vec(text_vec)
        text_index.add(
            np.asarray([norm_text_vec], dtype=np.float32)
        )

        # === 4. Extract and index audio features ===
        audio_available = False
        silent_audio = False
        audio_index_position = None
        dur_total = 0.0
        dur_active = 0.0

        try:
            audio_vec, dur_total, dur_active = self._extract_audio()

            audio_available = audio_vec is not None
            silent_audio = (
                audio_available and dur_active <= 0.0
            )

            # Do not add a missing-audio or silent-audio zero vector.
            if audio_available and not silent_audio:
                audio_index_position = audio_index.ntotal

                norm_audio = self.normalize_vec(audio_vec)
                audio_index.add(
                    np.asarray([norm_audio], dtype=np.float32)
                )

                print(
                    f"[AUDIO] Indexed audio for {self.video_path}; "
                    f"FAISS position={audio_index_position}"
                )
            else:
                print(
                    f"[AUDIO] Skipping audio index for {self.video_path}; "
                    f"audio_available={audio_available}, "
                    f"silent_audio={silent_audio}"
                )

        except Exception as audio_error:
            print(
                f"[WARNING] Audio processing failed for "
                f"{self.video_path}: {audio_error}"
            )
            audio_available = False
            silent_audio = True
            dur_total = 0.0
            dur_active = 0.0
            audio_index_position = None

        # === 5. Save FAISS indices ===
        self._save_faiss_index(video_index, video_index_path)
        self._save_faiss_index(text_index, text_index_path)
        self._save_faiss_index(audio_index, audio_index_path)

        print("Audio index count:", audio_index.ntotal)
        print("Video index count:", video_index.ntotal)
        print("Text index count:", text_index.ntotal)

        # === 6. Save metadata ===
        meta_path = os.path.join(self.output_dir, META_PATH)

        if os.path.exists(meta_path):
            with open(meta_path, "rb") as file:
                metadata = pickle.load(file)
        else:
            metadata = []

        duration = (
            scene_hashes[-1]["end"] if scene_hashes else 0.0
        )

        metadata.append({
            "filename": os.path.basename(self.video_path),
            "video_path": self.video_path,
            "num_scenes": len(scene_hashes),
            "duration": duration,
            "silent_audio": silent_audio,
            "audio_available": audio_available,
            "audio_duration": dur_total,
            "audio_active_duration": dur_active,
            "audio_index_position": audio_index_position
        })

        with open(meta_path, "wb") as file:
            pickle.dump(metadata, file)

        # === 7. Save video record to MongoDB ===
        doc = {
            "video_path": self.video_path,
            "duration": duration,
            "processed_at": datetime.utcnow(),
            "scene_hashes": scene_hashes,
            "audio_available": audio_available,
            "silent_audio": silent_audio,
            "audio_duration": dur_total,
            "audio_active_duration": dur_active,
            "audio_index_position": audio_index_position
        }

        self.db.videos.insert_one(doc)

        print(
            f"Inserted {self.video_path} with "
            f"{len(scene_hashes)} scenes."
        )
        print("FAISS indices and metadata saved.")

        elapsed = time.time() - start_time
        print(
            f"Time taken by {self.video_path}: {elapsed:.2f} seconds"
        )

    def compute_vectors(self):
        """
        Calculate vectors for a video without adding them to FAISS or MongoDB.

        The returned audio vector is zero-filled when audio is missing or
        silent. Callers should use audio_available/silent_audio to decide
        whether to compare that vector.
        """
        scenes = self._detect_scenes()

        hash_vectors = []
        texts = []

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"OpenCV could not open video: {self.video_path}"
            )

        try:
            for start, end in scenes:
                result = self._compute_frame(cap, start, end)

                if result and result[1] is not None:
                    _, hash_vec, text = result
                    hash_vectors.append(hash_vec)

                    if text:
                        texts.append(text)
        finally:
            cap.release()

        video_vec = (
            np.mean(hash_vectors, axis=0).astype(np.float32)
            if hash_vectors
            else np.zeros(VIDEO_DIM, dtype=np.float32)
        )

        text_vec = (
            self._text_embedding(texts)
            if texts
            else np.zeros(TEXT_DIM, dtype=np.float32)
        )

        audio_available = False
        silent_audio = True

        try:
            audio_vec, _, active_sec = self._extract_audio()

            audio_available = audio_vec is not None
            silent_audio = (
                not audio_available or active_sec <= 0.0
            )

            if not audio_available or silent_audio:
                audio_vec = np.zeros(AUDIO_DIM, dtype=np.float32)

        except Exception as audio_error:
            print(
                f"[WARNING] Audio extraction failed for "
                f"{self.video_path}: {audio_error}"
            )
            audio_vec = np.zeros(AUDIO_DIM, dtype=np.float32)
            audio_available = False
            silent_audio = True

        return {
            "video": self.normalize_vec(video_vec),
            "text": self.normalize_vec(text_vec),
            "audio": self.normalize_vec(audio_vec),
            "silent_audio": silent_audio,
            "audio_available": audio_available
        }
