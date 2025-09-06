"""
app.py
Flask application for multi-modal biometric authentication.
"""

from flask import Flask, render_template, request, redirect, url_for
from face_verification import get_face_embedding, calculate_face_similarity
from signature_verification import extract_signature_features, match_signatures
from utils import calculate_weighted_score
import os
import cv2
import pickle
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
USERS_DATA_FILE = 'users_data.pkl'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database for registered users
registered_users = {}
registered_signatures = {}

# Load previously registered users if they exist
def load_registered_users():
    global registered_users, registered_signatures
    if os.path.exists(USERS_DATA_FILE):
        try:
            with open(USERS_DATA_FILE, 'rb') as f:
                data = pickle.load(f)
                registered_users = data.get('faces', {})
                registered_signatures = data.get('signatures', {})
        except Exception as e:
            print(f"Error loading registered users: {e}")

# Save registered users
def save_registered_users():
    try:
        with open(USERS_DATA_FILE, 'wb') as f:
            data = {
                'faces': registered_users,
                'signatures': registered_signatures
            }
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving registered users: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            return "Name is required", 400

        # Capture image from camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Failed to capture image from camera", 500

        face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_face.jpg")
        cv2.imwrite(face_path, frame)

        try:
            # Capture face embedding
            face_embedding = get_face_embedding(face_path)
            registered_users[name] = face_embedding.tolist()  # Convert numpy array to list for serialization
            save_registered_users()
            return redirect(url_for('index'))
        except Exception as e:
            return str(e), 500

    return render_template('register_face.html')

@app.route('/register_signature', methods=['GET', 'POST'])
def register_signature():
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            return "Name is required", 400
            
        if 'signature_image' not in request.files:
            return "Missing signature image", 400

        signature_file = request.files['signature_image']
        if signature_file.filename == '':
            return "No selected file", 400

        signature_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_signature.jpg")
        signature_file.save(signature_path)
        
        try:
            # Extract signature features and store them
            keypoints, descriptors = extract_signature_features(signature_path)
            if descriptors is not None:
                registered_signatures[name] = signature_path
                save_registered_users()
                
            return redirect(url_for('index'))
        except Exception as e:
            return str(e), 500

    return render_template('register_signature.html')

@app.route('/verify_face', methods=['GET', 'POST'])
def verify_face():
    if request.method == 'GET':
        return render_template('verify_face.html')

    # POST request logic
    # Capture image from camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Failed to capture image from camera", 500

    face_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_face.jpg")
    cv2.imwrite(face_path, frame)

    try:
        # Face verification
        face_embedding = get_face_embedding(face_path)
        best_match = None
        best_score = 0
        
        for name, reg_emb_list in registered_users.items():
            reg_emb = np.array(reg_emb_list)  # Convert back to numpy array
            score = calculate_face_similarity(face_embedding, reg_emb)
            if score > best_score:
                best_score = score
                best_match = name

        result = "Verified" if best_score >= 0.65 else "Not Verified"
        return render_template('result_face.html', 
                               name=best_match if best_match else "Unknown", 
                               face_score=best_score, 
                               result=result)

    except Exception as e:
        return str(e), 500

@app.route('/verify_signature', methods=['GET', 'POST'])
def verify_signature():
    if request.method == 'GET':
        return render_template('verify_signature.html')
        
    if 'signature_image' not in request.files:
        return "Missing signature image", 400

    signature_file = request.files['signature_image']
    if signature_file.filename == '':
        return "No selected file", 400

    signature_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_signature.jpg")
    signature_file.save(signature_path)

    try:
        # Signature verification
        kp1, desc1 = extract_signature_features(signature_path)
        
        best_match = None
        best_score = 0
        
        for name, reg_signature_path in registered_signatures.items():
            kp2, desc2 = extract_signature_features(reg_signature_path)
            score = match_signatures(desc1, desc2)
            if score > best_score:
                best_score = score
                best_match = name

        result = "Verified" if best_score >= 0.65 else "Not Verified"
        return render_template('result_signature.html', 
                               name=best_match if best_match else "Unknown",
                               signature_score=best_score, 
                               result=result)

    except Exception as e:
        return str(e), 500

@app.route('/verify_combined', methods=['GET', 'POST'])
def verify_combined():
    if request.method == 'GET':
        return render_template('verify_combined.html')
        
    if 'face_image' not in request.files or 'signature_image' not in request.files:
        return "Missing files", 400

    face_file = request.files['face_image']
    signature_file = request.files['signature_image']

    if face_file.filename == '' or signature_file.filename == '':
        return "No selected file", 400

    face_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_face.jpg")
    signature_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_signature.jpg")

    face_file.save(face_path)
    signature_file.save(signature_path)

    try:
        # Face verification
        face_embedding = get_face_embedding(face_path)
        face_best_match = None
        face_best_score = 0
        
        for name, reg_emb_list in registered_users.items():
            reg_emb = np.array(reg_emb_list)
            score = calculate_face_similarity(face_embedding, reg_emb)
            if score > face_best_score:
                face_best_score = score
                face_best_match = name

        # Signature verification
        kp1, desc1 = extract_signature_features(signature_path)
        sig_best_match = None
        sig_best_score = 0
        
        for name, reg_signature_path in registered_signatures.items():
            kp2, desc2 = extract_signature_features(reg_signature_path)
            score = match_signatures(desc1, desc2)
            if score > sig_best_score:
                sig_best_score = score
                sig_best_match = name

        # Final score
        final_score = calculate_weighted_score(face_best_score, sig_best_score)
        result = "Verified" if final_score >= 0.65 else "Not Verified"

        return render_template('result_combined.html', 
                              face_name=face_best_match if face_best_match else "Unknown",
                              signature_name=sig_best_match if sig_best_match else "Unknown",
                              face_score=face_best_score, 
                              signature_score=sig_best_score,
                              final_score=final_score,
                              result=result)

    except Exception as e:
        return str(e), 500

# Load registered users at startup
load_registered_users()

if __name__ == '__main__':
    app.run(debug=True)
