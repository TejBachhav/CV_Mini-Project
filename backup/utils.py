"""
utils.py
Helper functions for similarity calculations and weighted score computation.
"""

def calculate_weighted_score(face_score, signature_score, face_weight=0.6, signature_weight=0.4):
    """
    Calculates the weighted score from face and signature scores.

    Args:
        face_score (float): Similarity score for face verification.
        signature_score (float): Similarity score for signature verification.
        face_weight (float): Weight for face score. Default is 0.6.
        signature_weight (float): Weight for signature score. Default is 0.4.

    Returns:
        float: Weighted score.
    """
    return (face_score * face_weight) + (signature_score * signature_weight)
