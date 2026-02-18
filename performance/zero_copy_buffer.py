
import numpy as np

class ZeroCopyBuffer:
    """
    Pre-allocates memory for NPU-visible frame data.
    Attempts to use AMD's unified memory space if available.
    Falls back to numpy array if XDNA driver not present.
    """
    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.use_unified = False
        
        # Try to import AMD XDNA runtime library
        try:
            # Fake/Hypothetical library name for XDNA buffer alloc
            import xdna_runtime
            self.buffer = xdna_runtime.allocate_buffer(shape, dtype)
            self.use_unified = True
        except ImportError:
            # Fallback
            self.buffer = np.zeros(shape, dtype=dtype)

    def write(self, data: np.ndarray):
        """Write data to the buffer."""
        if self.use_unified:
            # In real XDNA, this might be a direct memcpy or map
            # self.buffer.copy_from(data)
            np.copyto(self.buffer, data) # Mock unified behavior if it behaves like view
        else:
            np.copyto(self.buffer, data)

    def get_view(self) -> np.ndarray:
        """Get numpy-compatible view of the buffer."""
        return self.buffer

    def is_unified(self) -> bool:
        return self.use_unified
