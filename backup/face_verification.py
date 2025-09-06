"""
face_verification.py
Module for face verification using pre-trained models like FaceNet or DeepFace.
"""

import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    """
    Detects faces in an image and returns the count and image with face rectangles.
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        tuple: (face_count, image_with_faces)
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Make a copy of the image to draw on
    img_with_faces = image.copy()
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return len(faces), img_with_faces

def get_face_embedding(image_path_or_array):
    """
    Extracts a 128-D face embedding from the given image.

    Args:
        image_path_or_array: Path to the face image or a numpy array of the image.

    Returns:
        np.ndarray: 128-D face embedding vector.
    """
    try:
        # Check if image_path_or_array is already a numpy array
        if isinstance(image_path_or_array, np.ndarray):
            # Use the array directly
            embedding = DeepFace.represent(img_path=image_path_or_array, model_name='Facenet', detector_backend='opencv')[0]['embedding']
        else:
            # It's a file path
            embedding = DeepFace.represent(img_path=image_path_or_array, model_name='Facenet')[0]['embedding']
        
        return np.array(embedding)
    except Exception as e:
        raise ValueError(f"Error extracting face embedding: {e}")

def calculate_face_similarity(embedding1, embedding2):
    """
    Calculates the similarity between two face embeddings using cosine similarity.

    Args:
        embedding1 (np.ndarray): First face embedding.
        embedding2 (np.ndarray): Second face embedding.

    Returns:
        float: Similarity score between 0 and 1.
    """
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity
