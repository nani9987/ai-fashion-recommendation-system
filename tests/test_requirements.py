"""Test requirements.txt - Bug Fix Verification.

BUGS FIXED:
1. ✅ TensorFlow missing from requirements
2. ✅ tqdm missing (used in build_embeddings.py)
3. ✅ No version pinning
"""

import re
import pytest
from pathlib import Path


def test_tensorflow_in_requirements():
    """Bug Fix #1: TensorFlow must be in requirements."""
    with open("requirements.txt") as f:
        content = f.read()
    assert "tensorflow" in content.lower(), "TensorFlow must be in requirements.txt"


def test_tqdm_in_requirements():
    """Bug Fix #2: tqdm used in build_embeddings.py must be in requirements."""
    with open("requirements.txt") as f:
        content = f.read()
    assert "tqdm" in content.lower(), "tqdm must be in requirements.txt"


def test_all_packages_have_versions():
    """Bug Fix #3: All packages should have pinned versions."""
    with open("requirements.txt") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Each line should have package==version format
        assert "==" in line, f"Package '{line}' should have pinned version (e.g., package==1.0.0)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
