
import os
import sys
import time
import logging
import multiprocessing as mp
import numpy as np
import cv2

# Add project root to path for imports when running as worker
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules (must be importable)
try:
    from shield_face_pipeline import ShieldFacePipeline, FaceDetection
except ImportError:
    # Fallback/Mock for linting if paths are weird, but runtime should work
    pass

def _worker_loop(input_queue: mp.Queue, output_queue: mp.Queue, config: dict):
    """Worker process loop. Initializes pipeline and processes frames.
    
    Args:
        input_queue: Queue receiving (frame_id, frame_bgr).
        output_queue: Queue sending (frame_id, detections).
        config: Configuration dictionary for ShieldFacePipeline.
    """
    # Setup logging in worker
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger("PreprocessingWorker")
    _log.info("Worker process started (PID: %d)", os.getpid())

    # Initialize pipeline inside worker (avoid pickling issues)
    try:
        pipeline = ShieldFacePipeline(**config)
    except Exception as e:
        _log.error("Failed to initialize pipeline in worker: %s", e)
        return

    while True:
        try:
            item = input_queue.get()
            if item is None:
                break  # Sentinel to stop
            
            frame_id, frame = item
            
            # Run detection
            # Note: ShieldFacePipeline handles cropping/norm internally
            detections = pipeline.detect_faces(frame)
            
            # Detections (dataclasses) should be picklable
            output_queue.put((frame_id, detections))
            
        except Exception as e:
            _log.error("Worker exception: %s", e)
            output_queue.put((None, [])) # Signal error
    
    pipeline.release()
    _log.info("Worker process stopping")


class AsyncFacePipeline:
    """Multiprocessing wrapper for ShieldFacePipeline to bypass GIL.
    
    Offloads CPU-intensive face detection/alignment to a separate process.
    Mimics the API of ShieldFacePipeline.
    """

    def __init__(self, **kwargs):
        """Initialize worker process with given config kwargs."""
        self._input_queue = mp.Queue(maxsize=2)  # Limited buffer
        self._output_queue = mp.Queue(maxsize=2)
        self._worker = mp.Process(
            target=_worker_loop,
            args=(self._input_queue, self._output_queue, kwargs),
            daemon=True
        )
        self._worker.start()
        self._frame_counter = 0

    def detect_faces(self, frame: np.ndarray):
        """Send frame to worker and wait for results (Synchronous interface).
        
        Args:
            frame: BGR image.
        Returns:
            List of FaceDetection objects.
        """
        # Send frame
        self._frame_counter += 1
        current_id = self._frame_counter
        
        # In a real async pipeline, we'd feed queue and poll.
        # Here we block to maintain simplest "process_frame" contract
        # but execute in separate process (releasing GIL).
        self._input_queue.put((current_id, frame))
        
        # Wait for result
        try:
            res_id, detections = self._output_queue.get(timeout=2.0)
            if res_id != current_id:
                # Should not happen in synchronous usage
                return []
            return detections
        except Exception:
            return []

    def release(self):
        """Stop worker process."""
        self._input_queue.put(None)
        self._worker.join(timeout=1.0)
        if self._worker.is_alive():
            self._worker.terminate()
