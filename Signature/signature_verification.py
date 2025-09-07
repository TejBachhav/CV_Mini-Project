"""
signature_verification.py
Module for signature verification using OpenCV.
"""

import cv2
import numpy as np

def extract_signature_features(image_path):
    """
    Extracts keypoints and descriptors from a signature image using ORB.

    Args:
        image_path (str): Path to the signature image.

    Returns:
        tuple: Keypoints and descriptors.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image path or unable to read image.")

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_signatures(desc1, desc2):
    """
    Matches two sets of signature descriptors and calculates a similarity score.

    Args:
        desc1 (np.ndarray): Descriptors of the first signature.
        desc2 (np.ndarray): Descriptors of the second signature.

    Returns:
        float: Similarity score between 0 and 1.
    """
    if desc1 is None or desc2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    if not matches:
        return 0.0

    match_distances = [match.distance for match in matches]
    score = 1 - (np.mean(match_distances) / 100)
    return max(0.0, min(score, 1.0))
