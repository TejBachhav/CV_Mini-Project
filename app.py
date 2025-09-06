"""
app.py
Flask application for multi-modal biometric authentication with camera feed and multiple user support.
"""

from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, session
from face_verification import get_face_embedding, calculate_face_similarity, detect_faces
from signature_verification import extract_signature_features, match_signatures
from utils import calculate_weighted_score
import os
import cv2
import pickle
import numpy as np
import base64
import time
import datetime
from io import BytesIO
from PIL import Image
import uuid
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
USERS_DATA_FILE = 'users_data.pkl'
PROFILES_FOLDER = 'profiles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROFILES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database for registered users
registered_users = {}
registered_signatures = {}
user_profiles = {}

# Global variables for camera handling
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                logger.error("Failed to open camera")
                return None
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

# Load previously registered users if they exist
def load_registered_users():
    global registered_users, registered_signatures, user_profiles
    if os.path.exists(USERS_DATA_FILE):
        try:
            with open(USERS_DATA_FILE, 'rb') as f:
                data = pickle.load(f)
                registered_users = data.get('faces', {})
                registered_signatures = data.get('signatures', {})
                user_profiles = data.get('profiles', {})
            logger.info(f"Loaded {len(registered_users)} user profiles")
        except Exception as e:
            logger.error(f"Error loading registered users: {e}")

# Save registered users
def save_registered_users():
    try:
        with open(USERS_DATA_FILE, 'wb') as f:
            data = {
                'faces': registered_users,
                'signatures': registered_signatures,
                'profiles': user_profiles
            }
            pickle.dump(data, f)
        logger.info(f"Saved {len(registered_users)} user profiles")
    except Exception as e:
        logger.error(f"Error saving registered users: {e}")

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    """Video streaming generator function"""
    cam = get_camera()
    if cam is None:
        return
    
    try:
        while True:
            success, frame = cam.read()
            if not success:
                break
            else:
                # Check for multiple faces
                face_count, frame_with_faces = detect_faces(frame)
                
                # Add text based on face count
                if face_count == 0:
                    cv2.putText(frame_with_faces, "No face detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif face_count > 1:
                    cv2.putText(frame_with_faces, "Multiple faces detected!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame_with_faces, "Face detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame_with_faces)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        pass  # Don't release the camera here, it's shared

@app.route('/video_feed')
def video_feed():
    """Video streaming route for camera feed"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame')
def capture_frame():
    """Capture a single frame from the camera"""
    try:
        # Try to initialize a fresh camera connection each time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera in capture_frame")
            # Return a blank image with error text
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _, buffer = cv2.imencode('.jpg', error_img)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        # Read a frame
        success, frame = cap.read()
        # Release the camera immediately
        cap.release()
        if not success:
            logger.error("Failed to read frame in capture_frame")
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Failed to capture frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _, buffer = cv2.imencode('.jpg', error_img)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        # Check for multiple faces
        face_count, frame_with_faces = detect_faces(frame)
        # Add text based on face count
        if face_count == 0:
            cv2.putText(frame_with_faces, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif face_count > 1:
            cv2.putText(frame_with_faces, "Multiple faces detected!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame_with_faces, "Face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_with_faces)
        if not ret:
            logger.error("Failed to encode frame in capture_frame")
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Failed to encode frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _, buffer = cv2.imencode('.jpg', error_img)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        # Return the image as a response
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error in capture_frame: {e}")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error capturing frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        _, buffer = cv2.imencode('.jpg', error_img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/register_face', methods=['GET', 'POST'])
def register_face():
    if request.method == 'GET':
        return render_template('register_face.html')
    
    if request.method == 'POST':
        # For AJAX request to capture image
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                # Check if it's a data URL
                image_data = request.json.get('image')
                if not image_data or not image_data.startswith('data:image'):
                    return jsonify({"success": False, "error": "Invalid image data"})
                
                # Process image data URL
                image_data = image_data.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                
                # Convert PIL Image to OpenCV format
                image_np = np.array(image)
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Check face count
                face_count, _ = detect_faces(frame)
                
                if face_count == 0:
                    return jsonify({"success": False, "error": "No face detected"})
                elif face_count > 1:
                    return jsonify({"success": False, "error": "Multiple faces detected"})
                
                # Store the captured image temporarily in session
                session['temp_face_image'] = image_data
                
                return jsonify({"success": True})
            
            except Exception as e:
                logger.error(f"Error during face capture: {e}")
                return jsonify({"success": False, "error": str(e)})
        
        # Regular form submission to complete registration
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        if not name:
            return "Name is required", 400
        
        if 'temp_face_image' not in session:
            return "No captured image found. Please capture your face first.", 400
        
        # Get the captured image from session
        image_data = session.pop('temp_face_image')
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Convert PIL Image to OpenCV format
        image_np = np.array(image)
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Generate user ID
        user_id = str(uuid.uuid4())
        
        # Create user profile
        profile = {
            'name': name,
            'age': age,
            'gender': gender,
            'registration_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id
        }
        
        # Save the face image
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_face.jpg")
        cv2.imwrite(face_path, frame)
        profile['face_path'] = face_path
        
        try:
            # Capture face embedding
            face_embedding = get_face_embedding(frame)  # Use the frame directly
            
            # Store embedding and profile
            registered_users[user_id] = face_embedding.tolist()  # Convert numpy array to list for serialization
            user_profiles[user_id] = profile
            
            # Save to disk
            save_registered_users()
            
            return redirect(url_for('profile', user_id=user_id))
        
        except Exception as e:
            logger.error(f"Error during face registration: {e}")
            return str(e), 500

@app.route('/profile/<user_id>')
def profile(user_id):
    if user_id in user_profiles:
        profile = user_profiles[user_id]
        # Check if user has registered signature
        has_signature = user_id in registered_signatures
        return render_template('profile.html', profile=profile, has_signature=has_signature)
    else:
        return "User not found", 404

@app.route('/register_signature', methods=['GET', 'POST'])
def register_signature():
    if request.method == 'GET':
        # Get list of users who have registered faces but not signatures
        eligible_users = {uid: profile for uid, profile in user_profiles.items() 
                         if uid not in registered_signatures}
        return render_template('register_signature.html', users=eligible_users)
    
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if not user_id or user_id not in user_profiles:
            return "Invalid user selected", 400
            
        if 'signature_image' not in request.files:
            return "Missing signature image", 400

        signature_file = request.files['signature_image']
        if signature_file.filename == '':
            return "No selected file", 400

        signature_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_signature.jpg")
        signature_file.save(signature_path)
        
        try:
            # Extract signature features and store them
            keypoints, descriptors = extract_signature_features(signature_path)
            if descriptors is not None:
                registered_signatures[user_id] = signature_path
                save_registered_users()
                
            return redirect(url_for('profile', user_id=user_id))
        except Exception as e:
            logger.error(f"Error during signature registration: {e}")
            return str(e), 500

@app.route('/verify_face', methods=['GET', 'POST'])
def verify_face():
    if request.method == 'GET':
        return render_template('verify_face.html', users=user_profiles)
    
    # AJAX request: receives image captured client-side (browser) as base64
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            # Check if it's a data URL from client-side capture
            image_data = request.json.get('image')
            if not image_data or not image_data.startswith('data:image'):
                return jsonify({"success": False, "error": "Invalid image data"})
            # Log for clarity
            logger.info("Received face image from client-side capture for verification.")
            # Process image data URL
            image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            # Convert PIL Image to OpenCV format
            image_np = np.array(image)
            frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            # Check face count
            face_count, _ = detect_faces(frame)
            if face_count == 0:
                return jsonify({"success": False, "error": "No face detected"})
            elif face_count > 1:
                return jsonify({"success": False, "error": "Multiple faces detected"})
            try:
                # Face verification
                face_embedding = get_face_embedding(frame)  # Use the frame directly
                best_match_id = None
                best_score = 0
                
                for user_id, reg_emb_list in registered_users.items():
                    reg_emb = np.array(reg_emb_list)  # Convert back to numpy array
                    score = calculate_face_similarity(face_embedding, reg_emb)
                    if score > best_score:
                        best_score = score
                        best_match_id = user_id
                
                result = "Verified" if best_score >= 0.65 else "Not Verified"
                
                # Get user profile if verified
                profile = None
                if best_match_id and best_match_id in user_profiles:
                    profile = user_profiles[best_match_id]
                
                response_data = {
                    "success": True,
                    "result": result,
                    "score": float(best_score),
                    "profile": profile,
                    "threshold": 0.65
                }
                
                return jsonify(response_data)
            
            except Exception as e:
                logger.error(f"Error during face verification: {e}")
                return jsonify({"success": False, "error": str(e)})
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({"success": False, "error": str(e)})

    # If it's a regular POST but not AJAX, reject
    return "Invalid request", 400

@app.route('/process_verification', methods=['POST'])
def process_verification():
    """Process verification for both face and signature"""
    if not request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return "Invalid request", 400
    
    try:
        # All image capture is now handled client-side; server only processes uploaded image
        # Get the JSON data
        data = request.json
        verification_type = data.get('type')
        username = data.get('username')
        image_data = data.get('image')
        if not username or not image_data:
            logger.error("Missing required data in process_verification")
            return jsonify({"success": False, "error": "Missing required data"})
        # For direct user verification, use the user ID directly
        user_id = username  # The username select now sends the user ID directly
        if not user_id in user_profiles:
            logger.error(f"User ID {user_id} not found in process_verification")
            return jsonify({"success": False, "error": "User not found"})
        # Process image data URL
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image_np = np.array(image)
            # Check if image is RGB (from canvas) or RGBA (sometimes from canvas)
            if image_np.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error processing image data: {e}")
            return jsonify({"success": False, "error": f"Error processing image: {str(e)}"})
        # Process based on verification type
        if verification_type == 'face':
            # Check face count
            try:
                face_count, _ = detect_faces(frame)
                if face_count == 0:
                    return jsonify({"success": False, "error": "No face detected in the image"})
                elif face_count > 1:
                    return jsonify({"success": False, "error": "Multiple faces detected in the image"})
            except Exception as e:
                logger.error(f"Error in face detection: {e}")
                return jsonify({"success": False, "error": f"Error detecting faces: {str(e)}"})
            try:
                # Get user's stored face embedding
                if user_id not in registered_users:
                    return jsonify({"success": False, "error": "User has no registered face"})
                stored_emb = np.array(registered_users[user_id])
                # Get current face embedding
                current_emb = get_face_embedding(frame)
                # Calculate similarity
                score = calculate_face_similarity(current_emb, stored_emb)
                threshold = 0.65
                logger.info(f"Face verification score: {score}, threshold: {threshold}")
                # Return results
                return jsonify({
                    "success": True,
                    "verification_success": bool(score >= threshold),
                    "face_score": float(score),
                    "threshold": float(threshold)
                })
            except Exception as e:
                logger.error(f"Error during face verification: {e}")
                return jsonify({"success": False, "error": f"Error during verification: {str(e)}"})
        # Add other verification types if needed
        return jsonify({"success": False, "error": "Invalid verification type"})
    except Exception as e:
        logger.error(f"Error in process_verification: {e}")
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"})

@app.route('/verify_signature', methods=['GET', 'POST'])
def verify_signature():
    if request.method == 'GET':
        # Pass user_profiles as 'users' to the template
        return render_template('verify_signature.html', users=user_profiles)
        
    if request.method == 'POST':
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
            
            best_match_id = None
            best_score = 0
            
            for user_id, reg_signature_path in registered_signatures.items():
                kp2, desc2 = extract_signature_features(reg_signature_path)
                score = match_signatures(desc1, desc2)
                if score > best_score:
                    best_score = score
                    best_match_id = user_id
            
            result = "Verified" if best_score >= 0.65 else "Not Verified"
            
            # Get user profile if verified
            profile = None
            if best_match_id and best_match_id in user_profiles:
                profile = user_profiles[best_match_id]
            
            return render_template('result_signature.html', 
                                name=profile['name'] if profile else "Unknown",
                                user_id=best_match_id,
                                profile=profile,
                                signature_score=best_score, 
                                result=result)

        except Exception as e:
            logger.error(f"Error during signature verification: {e}")
            return str(e), 500

@app.route('/verify_combined', methods=['GET', 'POST'])
def verify_combined():
    if request.method == 'GET':
        return render_template('verify_combined.html')
        
    if request.method == 'POST':
        # For AJAX request to capture face image
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                # Check if it's a data URL
                image_data = request.json.get('image')
                if not image_data or not image_data.startswith('data:image'):
                    return jsonify({"success": False, "error": "Invalid image data"})
                
                # Process image data URL
                image_data = image_data.split(',')[1]
                session['temp_face_image_verify'] = image_data
                
                return jsonify({"success": True})
            
            except Exception as e:
                logger.error(f"Error during face capture: {e}")
                return jsonify({"success": False, "error": str(e)})
        
        # Regular form submission to complete verification
        if 'signature_image' not in request.files:
            return "Missing signature image", 400
            
        signature_file = request.files['signature_image']
        if signature_file.filename == '':
            return "No selected file", 400
            
        if 'temp_face_image_verify' not in session:
            return "No captured face image. Please capture your face first.", 400
        
        # Process the face image
        image_data = session.pop('temp_face_image_verify')
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Convert PIL Image to OpenCV format
        image_np = np.array(image)
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Save the signature image
        signature_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_signature.jpg")
        signature_file.save(signature_path)

        try:
            # Face verification
            face_embedding = get_face_embedding(frame)
            face_best_match_id = None
            face_best_score = 0
            
            for user_id, reg_emb_list in registered_users.items():
                reg_emb = np.array(reg_emb_list)
                score = calculate_face_similarity(face_embedding, reg_emb)
                if score > face_best_score:
                    face_best_score = score
                    face_best_match_id = user_id

            # Signature verification
            kp1, desc1 = extract_signature_features(signature_path)
            sig_best_match_id = None
            sig_best_score = 0
            
            for user_id, reg_signature_path in registered_signatures.items():
                kp2, desc2 = extract_signature_features(reg_signature_path)
                score = match_signatures(desc1, desc2)
                if score > sig_best_score:
                    sig_best_score = score
                    sig_best_match_id = user_id

            # Get user profiles
            face_profile = user_profiles.get(face_best_match_id) if face_best_match_id else None
            sig_profile = user_profiles.get(sig_best_match_id) if sig_best_match_id else None
            
            # Final score
            final_score = calculate_weighted_score(face_best_score, sig_best_score)
            
            # Check if the same person is identified by both methods
            same_person = face_best_match_id is not None and face_best_match_id == sig_best_match_id
            
            # If same person is identified with good scores, it's a strong verification
            strong_verification = same_person and final_score >= 0.7
            
            # Final result
            if strong_verification:
                result = "Strongly Verified"
            elif final_score >= 0.65:
                result = "Verified"
            else:
                result = "Not Verified"

            return render_template('result_combined.html', 
                                face_name=face_profile['name'] if face_profile else "Unknown",
                                signature_name=sig_profile['name'] if sig_profile else "Unknown",
                                face_profile=face_profile,
                                sig_profile=sig_profile,
                                same_person=same_person,
                                face_score=face_best_score, 
                                signature_score=sig_best_score,
                                final_score=final_score,
                                result=result)

        except Exception as e:
            logger.error(f"Error during combined verification: {e}")
            return str(e), 500

@app.route('/all_users')
def all_users():
    """Show all registered users"""
    return render_template('all_users.html', users=user_profiles)

@app.route('/clean_up', methods=['GET', 'POST'])
def clean_up():
    """Clean up and release resources before shutdown"""
    release_camera()
    return "Resources released"

@app.route('/process_registration', methods=['POST'])
def process_registration():
    if not request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({"success": False, "error": "Invalid request"}), 400

    data = request.get_json()
    user_id = data.get('user_id')
    username = data.get('username')
    age = data.get('age')
    gender = data.get('gender')
    image_data = data.get('image')

    if not image_data or not username or not age or not gender or not user_id:
        return jsonify({"success": False, "error": "Missing image, username, user_id, age, or gender"}), 400

    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Face detection
        face_count, _ = detect_faces(frame)
        if face_count == 0:
            return jsonify({"success": False, "error": "No face detected"})
        elif face_count > 1:
            return jsonify({"success": False, "error": "Multiple faces detected"})

        # Save image paths
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_face.jpg")
        cv2.imwrite(face_path, frame)
        profile_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_id}_profile.jpg")
        cv2.imwrite(profile_pic_path, frame)

        # Get face embedding
        face_embedding = get_face_embedding(frame)
        profile = {
            'name': username,
            'age': age,
            'gender': gender,
            'registration_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'face_path': face_path,
            'profile_pic': profile_pic_path
        }
        registered_users[user_id] = face_embedding.tolist()
        user_profiles[user_id] = profile
        save_registered_users()

        return jsonify({"success": True, "message": "Face registered successfully!"})
    except Exception as e:
        logger.error(f"Error in process_registration: {e}")
        return jsonify({"success": False, "error": str(e)})

# Load registered users at startup
load_registered_users()

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        release_camera()  # Make sure camera is released when app shuts down
