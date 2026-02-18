
import numpy as np
import cv2
from collections import deque

class TemporalConsistencyAnalyzer:
    """
    Analyzes frame sequences for temporal artifacts common in deepfakes:
    flickering, temporal discontinuity (embedding jumps), inconsistent lighting.
    """
    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self.frame_buffer = deque(maxlen=window_size)
        self.embedding_buffer = deque(maxlen=window_size)
        self.lighting_buffer = deque(maxlen=window_size)

    def add_frame(self, frame: np.ndarray, embedding: np.ndarray):
        """Add frame and its neural embedding to buffer.
        
        Args:
            frame: RGB or BGR image (uint8).
            embedding: (N,) vector from face recognition model or similar.
        """
        self.frame_buffer.append(frame)
        self.embedding_buffer.append(embedding)
        
        # Calculate simple lighting metric (average intensity)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.lighting_buffer.append(np.mean(gray))

    def _cosine_similarity(self, a, b):
        if a is None or b is None:
            return 1.0 # Skip
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def analyze(self) -> dict:
        """
        Analyze the buffered frames for temporal anomalies.
        
        Returns:
            dict with scores and detection flags.
        """
        if len(self.frame_buffer) < self.window_size:
            return {"is_ready": False, "temporal_score": 0.0}

        # 1. Embedding Consistency (Cosine Similarity)
        # Calculate similarity between adjacent embeddings
        similarities = []
        embeddings = list(self.embedding_buffer)
        for i in range(len(embeddings) - 1):
             sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
             similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        embedding_consistency = float(avg_similarity)

        # 2. Lighting Consistency (Intensity Variance)
        # Deepfakes often have flickering lighting
        lighting_std = np.std(list(self.lighting_buffer))
        # High variance suggests flickering. Normalize to [0, 1] heuristic?
        # Typically lighting changes slowly. Rapid variance > Threshold is suspicious.
        lighting_consistency = max(0.0, 1.0 - (lighting_std / 50.0)) # Heuristic scaling

        # 3. Pixel-level Flickering (Mean Absolute Difference)
        # Only feasible if face is aligned perfectly.
        # We'll skip pixel-diff if we don't have aligned crops in buffer.
        # Assuming frame inputs are aligned face crops.
        flicker_score = 1.0
        diffs = []
        frames = list(self.frame_buffer)
        for i in range(len(frames) - 1):
            f1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            f2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            diff = np.mean(np.abs(f1.astype(float) - f2.astype(float)))
            diffs.append(diff)
        
        avg_diff = np.mean(diffs) if diffs else 0.0
        flickering_detected = avg_diff > 15.0 # Threshold for rapid change
        
        # Aggregate Score
        # If consistency is low, score drops.
        # If flickering detected, score drops.
        temporal_score = (embedding_consistency * 0.4 + lighting_consistency * 0.4 + (0.0 if flickering_detected else 0.2))
        
        return {
            "temporal_score": round(temporal_score, 3),
            "flickering_detected": bool(flickering_detected),
            "embedding_consistency": round(embedding_consistency, 3),
            "lighting_consistency": round(lighting_consistency, 3),
            "is_ready": True
        }
