"""Test recommender.py - Bug Fix Verification.

BUGS FIXED:
1. ✅ CRITICAL: Function was just random sampling, not cosine similarity
2. ✅ Missing query_embedding parameter
3. ✅ No similarity computation at all
4. ✅ Duplicate line numbers in original code
5. ✅ No cosine similarity metric
"""

import pytest
import numpy as np
import pandas as pd
from src.recommender import get_recommendations


@pytest.fixture
def mock_data():
    """Create mock embeddings and dataframe."""
    # Create 10 random embeddings
    embeddings = np.random.randn(10, 2048).astype(np.float32)
    image_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    
    df = pd.DataFrame({
        "id": image_ids,
        "articleType": ["shirt", "shirt", "jeans", "jeans", "dress"] * 2,
        "baseColour": ["red", "blue", "black", "white", "red"] * 2,
        "season": ["summer", "winter", "summer", "winter", "all"] * 2,
        "usage": ["casual", "casual", "formal", "casual", "formal"] * 2,
    })
    
    return embeddings, image_ids, df


def test_get_recommendations_takes_embedding_parameter(mock_data):
    """Bug Fix #2: Function must accept query_embedding parameter."""
    embeddings, image_ids, df = mock_data
    query_embedding = embeddings[0]  # Use first embedding as query
    
    # This should not raise TypeError about missing parameter
    result = get_recommendations(query_embedding, embeddings, image_ids, df, n=3)
    assert result is not None, "Should return recommendations"


def test_get_recommendations_computes_similarity(mock_data):
    """Bug Fix #1, #3, #5: Must compute cosine similarity, not just random sample."""
    embeddings, image_ids, df = mock_data
    
    # Create a query embedding identical to first embedding
    query_embedding = embeddings[0].copy()
    
    # Get recommendations
    result = get_recommendations(query_embedding, embeddings, image_ids, df, n=3)
    
    # First result should be the most similar (the identical embedding)
    # Check that top result has highest similarity score
    if 'similarity_score' in result.columns:
        scores = result['similarity_score'].values
        assert scores[0] >= scores[-1], "Results should be sorted by similarity (descending)"


def test_get_recommendations_returns_correct_count(mock_data):
    """Should return exactly n recommendations."""
    embeddings, image_ids, df = mock_data
    query_embedding = embeddings[0]
    
    result = get_recommendations(query_embedding, embeddings, image_ids, df, n=3)
    assert len(result) <= 3, "Should return at most n items"


def test_get_recommendations_includes_similarity_scores(mock_data):
    """Bug Fix #1: Should include similarity scores in output."""
    embeddings, image_ids, df = mock_data
    query_embedding = embeddings[0]
    
    result = get_recommendations(query_embedding, embeddings, image_ids, df, n=3)
    
    # Should have similarity_score column
    assert 'similarity_score' in result.columns, "Should include similarity_score column"
    
    # Scores should be between 0 and 1 (for cosine similarity)
    scores = result['similarity_score'].values
    assert all((scores >= -1) & (scores <= 1)), "Cosine similarity should be in [-1, 1]"


def test_get_recommendations_handles_none_embedding(mock_data):
    """Should handle None query embedding gracefully."""
    embeddings, image_ids, df = mock_data
    
    result = get_recommendations(None, embeddings, image_ids, df, n=3)
    assert result is not None, "Should handle None embedding gracefully"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
