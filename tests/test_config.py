"""Test config.py - Bug Fix Verification.

BUGS FIXED:
1. ✅ Hardcoded relative paths → Using Path objects
2. ✅ No validation if paths exist
3. ✅ Missing embeddings config
"""

import pytest
from pathlib import Path
from src.config import BASE_DIR, DATA_DIR, IMAGE_DIR, CSV_PATH, EMBEDDINGS_DIR


def test_paths_are_pathlib_objects():
    """Bug Fix #1: Verify all paths use pathlib.Path, not strings."""
    assert isinstance(BASE_DIR, Path), "BASE_DIR should be Path object"
    assert isinstance(DATA_DIR, Path), "DATA_DIR should be Path object"
    assert isinstance(IMAGE_DIR, Path), "IMAGE_DIR should be Path object"
    assert isinstance(CSV_PATH, Path), "CSV_PATH should be Path object"
    assert isinstance(EMBEDDINGS_DIR, Path), "EMBEDDINGS_DIR should be Path object"


def test_paths_are_absolute():
    """Bug Fix #1: Verify paths are absolute (work from any directory)."""
    assert BASE_DIR.is_absolute(), "BASE_DIR should be absolute path"
    assert DATA_DIR.is_absolute(), "DATA_DIR should be absolute path"
    assert CSV_PATH.is_absolute(), "CSV_PATH should be absolute path"


def test_csv_path_points_to_styles_csv():
    """Bug Fix #2: Verify CSV path is correct."""
    assert CSV_PATH.name == "styles.csv", "CSV_PATH should point to styles.csv"
    assert CSV_PATH.parent.name == "data", "CSV should be in data/ directory"


def test_embeddings_config_exists():
    """Bug Fix #3: Verify embeddings config is defined."""
    from src.config import EMBEDDING_DIM, MODEL_NAME
    assert EMBEDDING_DIM == 2048, "Should be ResNet50 dimension"
    assert MODEL_NAME == "ResNet50", "Should use ResNet50"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
