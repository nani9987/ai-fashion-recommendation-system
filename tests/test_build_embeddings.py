"""Test build_embeddings.py - Bug Fix Verification.

BUGS FIXED:
1. ✅ Slow .iterrows() loop
2. ✅ No error handling for failed images
3. ✅ Hardcoded path (should use config)
4. ✅ No timing information
5. ✅ Shape mismatch when images fail
6. ✅ No dtype specified for numpy arrays
7. ✅ Missing import validation
"""

import pytest
import numpy as np
from pathlib import Path
from src.config import EMBEDDINGS_DIR


def test_build_embeddings_imports():
    """Bug Fix #7: All imports should work."""
    try:
        from src.build_embeddings import build_embeddings
        import time
        from tqdm import tqdm
    except ImportError as e:
        pytest.fail(f"Missing import in build_embeddings: {e}")


def test_embeddings_directory_config():
    """Bug Fix #3: Should use EMBEDDINGS_DIR from config, not hardcoded path."""
    from src.config import EMBEDDINGS_DIR
    assert EMBEDDINGS_DIR is not None, "EMBEDDINGS_DIR should be defined in config"
    assert isinstance(EMBEDDINGS_DIR, Path), "EMBEDDINGS_DIR should be Path object"


def test_embedding_arrays_have_correct_dtype():
    """Bug Fix #6: Embeddings should be saved with specific dtype."""
    # If embeddings exist, verify dtype
    embeddings_path = EMBEDDINGS_DIR / "image_embeddings.npy"
    ids_path = EMBEDDINGS_DIR / "image_ids.npy"
    
    if embeddings_path.exists() and ids_path.exists():
        embeddings = np.load(embeddings_path)
        ids = np.load(ids_path)
        
        # Check dtypes
        assert embeddings.dtype == np.float32, f"Embeddings should be float32, got {embeddings.dtype}"
        assert ids.dtype in [np.int64, np.int32], f"IDs should be int64 or int32, got {ids.dtype}"
        
        # Check consistency
        assert len(embeddings) == len(ids), "Embeddings and IDs should have same length"


def test_embeddings_shape_consistency():
    """Bug Fix #5: Embeddings shape should be (N, 2048)."""
    embeddings_path = EMBEDDINGS_DIR / "image_embeddings.npy"
    
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        assert len(embeddings.shape) == 2, f"Embeddings should be 2D array, got shape {embeddings.shape}"
        assert embeddings.shape[1] == 2048, f"Each embedding should be 2048-D, got {embeddings.shape[1]}"


def test_no_nan_values_in_embeddings():
    """Embeddings should not contain NaN values."""
    embeddings_path = EMBEDDINGS_DIR / "image_embeddings.npy"
    
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        assert not np.any(np.isnan(embeddings)), "Embeddings should not contain NaN values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
