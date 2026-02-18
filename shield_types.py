
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class FaceResult:
    """Analysis result for a single face."""
    bbox: Tuple[int, int, int, int]
    state: str
    neural_confidence: float
    ear_value: float
    ear_reliability: str
    texture_score: float
    texture_explanation: str
    tier_results: Tuple[str, str, str]
    occlusion_score: float
    advanced_info: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class EngineResult:
    """Aggregate result for a processed frame."""
    frame: Optional[np.ndarray]
    state: str                    # overall system state (e.g., 'REAL', 'NO_FACE')
    face_results: List[FaceResult]
    fps: float
    timing_breakdown: dict
    camera_health: dict
