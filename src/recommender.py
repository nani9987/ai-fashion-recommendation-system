import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(query_embedding, embeddings, image_ids, df, n=6):
    """
    Get top-n recommendations based on cosine similarity of CNN features.
    
    Args:
        query_embedding: (2048,) feature vector from query image
        embeddings: (M, 2048) array of all image embeddings
        image_ids: (M,) array of image IDs corresponding to embeddings
        df: DataFrame with metadata
        n: Number of recommendations to return
    
    Returns:
        DataFrame with top-n similar items (sorted by similarity score)
    """
    if query_embedding is None or len(embeddings) == 0:
        print("Warning: Returning random items (no valid embeddings)")
        return df.head(n)
    
    # Compute cosine similarity: (1, M) array
    similarities = cosine_similarity(
        [query_embedding], 
        embeddings
    )[0]  # Shape: (M,)
    
    # Get indices of top-n similar items (descending order)
    top_indices = np.argsort(-similarities)[:n]
    top_ids = image_ids[top_indices]
    
    # Return matching rows from dataframe
    recommendations = df[df['id'].isin(top_ids)].copy()
    
    # Add similarity scores for reference
    recommendations['similarity_score'] = recommendations['id'].map(
        {img_id: float(sim) for img_id, sim in zip(image_ids[top_indices], similarities[top_indices])}
    )
    
    return recommendations.sort_values('similarity_score', ascending=False)
