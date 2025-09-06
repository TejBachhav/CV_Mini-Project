# Multi-Modal Biometric Authentication System

This project implements a biometric authentication system using multiple factors: face recognition and signature verification. The system provides a web interface built with Flask for user registration and verification.

## Features

- **Face Verification**: Using deep learning models for face embedding extraction and comparison
- **Signature Verification**: Using computer vision techniques for signature feature extraction and matching
- **Multi-Modal Authentication**: Combining scores from both face and signature verification
- **Live Camera Feed**: Real-time face capture during registration and verification
- **User Profile Management**: Support for multiple user profiles
- **Face Detection**: Ensures only one person is present in the camera view

## Project Structure

```
.
├── app.py                     # Flask application
├── face_verification.py       # Face embedding and verification functionality
├── signature_verification.py  # Signature feature extraction and matching
├── utils.py                   # Utility functions
├── uploads/                   # Directory for storing uploaded images
└── templates/                 # HTML templates for the web interface
    ├── index.html                # Home page
    ├── register.html             # Registration options page
    ├── register_face.html        # Face registration page
    ├── register_signature.html   # Signature registration page
    ├── verify_face.html          # Face verification page
    ├── verify_signature.html     # Signature verification page
    ├── verify_combined.html      # Combined verification page
    ├── result_face.html          # Face verification result page
    ├── result_signature.html     # Signature verification result page
    ├── result_combined.html      # Combined verification result page
    ├── profile.html              # User profile page
    └── all_users.html            # All users listing page
```

## Requirements

- Python 3.7+
- Flask
- OpenCV
- DeepFace (or FaceNet)
- NumPy
- scikit-learn

## Setup and Installation

1. **Clone the repository**:
```
git clone <repository-url>
cd multi-modal-biometric-auth
```

2. **Create and activate a virtual environment**:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```
pip install -r requirements.txt
```

4. **Run the application**:
```
python app.py
```

5. **Access the web interface**:
Open your browser and navigate to http://127.0.0.1:5000/

## Usage

1. **Registration**:
   - Navigate to the registration page
   - Enter your username
   - Capture your face using the camera
   - Upload your signature image

2. **Verification**:
   - Choose the verification method (face, signature, or combined)
   - Follow the on-screen instructions
   - View the verification results

## Configuration

You can adjust the following parameters in the `app.py` file:
- `FACE_WEIGHT` (default: 0.6): Weight for face verification score
- `SIGNATURE_WEIGHT` (default: 0.4): Weight for signature verification score
- `SIMILARITY_THRESHOLD` (default: 0.7): Threshold for successful verification

## Acknowledgments

This project was developed as part of a computer vision mini-project.
