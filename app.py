from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import numpy as np
import cv2
from cryptography.fernet import Fernet
import json
from sklearn.neighbors import KNeighborsClassifier
import face_recognition
import time
from datetime import datetime
import os
import firebase_admin
from firebase_admin import credentials, auth, firestore
from dotenv import load_dotenv
import base64
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Initialize Firebase Admin
cred = credentials.Certificate('firebase-service-account.json')  # You'll need to add your Firebase service account key
firebase_admin.initialize_app(cred)
db = firestore.client()

# JWT Configuration
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this in production
jwt = JWTManager(app)

# Encryption Key - Make it persistent
def get_or_create_key():
    key_file = 'encryption.key'
    if os.path.exists(key_file):
        with open(key_file, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        return key

key = get_or_create_key()
cipher = Fernet(key)

# Database to store encrypted features and labels
database = {
    "features": [],
    "labels": []
}

# Initialize face recognition
knn = KNeighborsClassifier(n_neighbors=1)

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Face Biometric System API is running",
        "endpoints": {
            "auth": {
                "signup": "/api/auth/signup",
                "login": "/api/auth/login"
            },
            "face": {
                "register": "/api/face/register",
                "extract": "/api/face/extract",
                "optimize": "/api/face/optimize",
                "match": "/api/face/match"
            }
        }
    })

# ==============================
# Authentication
# ==============================
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    name = data.get("name")

    if not email or not password or not name:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Create user in Firebase Authentication
        user = auth.create_user(
            email=email,
            password=password,
            display_name=name
        )

        # Store additional user data in Firestore
        db.collection('users').document(user.uid).set({
            'name': name,
            'email': email,
            'created_at': datetime.now()
        })

        # Create JWT token
        access_token = create_access_token(identity=user.uid)

        return jsonify({
            "token": access_token,
            "user": {
                "id": user.uid,
                "email": email,
                "name": name
            }
        })
    except auth.EmailAlreadyExistsError:
        return jsonify({"error": "Email already registered"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Sign in with Firebase
        user = auth.get_user_by_email(email)
        
        # Create JWT token
        access_token = create_access_token(identity=user.uid)

        # Get user data from Firestore
        user_data = db.collection('users').document(user.uid).get().to_dict()

        return jsonify({
            "token": access_token,
            "user": {
                "id": user.uid,
                "email": email,
                "name": user_data.get('name')
            }
        })
    except auth.UserNotFoundError:
        return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Face Recognition API
# ==============================
@app.route("/api/face/register", methods=["POST"])
def register_face():
    try:
        print("Received face registration request")  # Debug log
        
        # Verify Firebase token
        token = request.headers.get("Authorization")
        print(f"Authorization header: {token}")  # Debug log
        
        if not token:
            print("No token provided")  # Debug log
            return jsonify({"error": "No token provided"}), 401
        
        # Verify token using Firebase
        decoded_token = verify_firebase_token(token)
        if not decoded_token:
            print("Invalid token")  # Debug log
            return jsonify({"error": "Invalid token"}), 401
        
        user_id = decoded_token["uid"]
        print(f"User ID: {user_id}")  # Debug log
        
        # Get image data
        if "image" not in request.files:
            print("No image in request.files")  # Debug log
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files["image"]
        print(f"Image file received: {image_file.filename}")  # Debug log
        image_data = image_file.read()
        
        # Convert image data to numpy array for face detection
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")  # Debug log
            return jsonify({"error": "Failed to decode image"}), 400
            
        print(f"Image shape: {img.shape}")  # Debug log
        
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check if face is detected
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            print("No face detected in image")  # Debug log
            return jsonify({"error": "No face detected in image. Please make sure your face is clearly visible."}), 400
            
        print(f"Found {len(face_locations)} faces in image")  # Debug log
        
        # Extract and process features
        print("Attempting to extract face features")  # Debug log
        features = extract_face_features(image_data)
        if features is None:
            print("Failed to extract face features")  # Debug log
            return jsonify({"error": "Failed to extract face features. Please try again with a clearer image."}), 400
        
        print(f"Features extracted successfully. Shape: {features.shape}")  # Debug log
        print(f"Features values range: [{np.min(features)}, {np.max(features)}]")  # Debug log
        
        # Store raw features without optimization
        print("Encrypting features")  # Debug log
        encrypted_features = encrypt_face_features(features)
        
        # Store in Firestore
        print("Storing features in Firestore")
        user_ref = db.collection("users").document(user_id)
        user_ref.update({
            "face_features": encrypted_features,
            "has_face_registered": True,
            "face_registration_date": datetime.now().isoformat()
        })
        
        # Verify the update
        updated_user = user_ref.get().to_dict()
        print(f"Updated user data: {updated_user}")  # Debug log
        
        # Verify the stored features can be retrieved and decrypted
        try:
            stored_features = decrypt_face_features(base64.b64decode(encrypted_features))
            print(f"Verified stored features shape: {stored_features.shape}")  # Debug log
            print(f"Verified stored features range: [{np.min(stored_features)}, {np.max(stored_features)}]")  # Debug log
        except Exception as e:
            print(f"Error verifying stored features: {str(e)}")  # Debug log
        
        print("Face registration completed successfully")  # Debug log
        return jsonify({
            "message": "Face registered successfully",
            "debug_info": {
                "features_shape": features.shape,
                "features_range": [float(np.min(features)), float(np.max(features))],
                "user_id": user_id
            }
        })
    
    except Exception as e:
        print(f"Error in register_face: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

@app.route("/api/face/extract", methods=["POST"])
def extract_face_features():
    try:
        print("Starting face feature extraction...")  # Debug log
        
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            print("No authorization header found")  # Debug log
            return jsonify({"error": "No authorization token provided"}), 401

        # Verify the Firebase token
        decoded_token = verify_firebase_token(auth_header)
        if not decoded_token:
            print("Token verification failed")  # Debug log
            return jsonify({"error": "Invalid token"}), 401

        user_id = decoded_token['uid']
        print(f"Processing request for user: {user_id}")  # Debug log

        # Get image from request
        if "image" not in request.files:
            print("No image in request.files")  # Debug log
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        print(f"Image file received: {image_file.filename}")  # Debug log
        image_data = image_file.read()

        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("Failed to decode image")  # Debug log
            return jsonify({"error": "Failed to decode image"}), 400

        print(f"Image shape: {img.shape}")  # Debug log

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Converted image to grayscale")  # Debug log

        # Detect faces
        face_locations = face_recognition.face_locations(gray)
        if not face_locations:
            print("No faces detected in image")  # Debug log
            return jsonify({"error": "No face detected in the image"}), 400

        print(f"Found {len(face_locations)} face(s) in the image")  # Debug log

        # Get face encodings
        face_encodings = face_recognition.face_encodings(img, face_locations)
        if not face_encodings:
            print("Failed to get face encodings")  # Debug log
            return jsonify({"error": "Failed to extract face features"}), 400

        # Get facial landmarks
        face_landmarks_list = face_recognition.face_landmarks(img, face_locations)
        if not face_landmarks_list:
            print("Failed to get face landmarks")  # Debug log
            return jsonify({"error": "Failed to extract face landmarks"}), 400

        # Create a copy of the grayscale image for visualization
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw facial landmarks with triangles
        for face_landmarks in face_landmarks_list:
            # Draw chin
            for i in range(len(face_landmarks['chin']) - 1):
                pt1 = face_landmarks['chin'][i]
                pt2 = face_landmarks['chin'][i + 1]
                cv2.line(processed_img, pt1, pt2, (0, 255, 0), 2)
                # Draw triangle
                if i < len(face_landmarks['chin']) - 2:
                    pt3 = face_landmarks['chin'][i + 2]
                    cv2.line(processed_img, pt2, pt3, (0, 255, 0), 2)
                    cv2.line(processed_img, pt3, pt1, (0, 255, 0), 2)

            # Draw other facial features
            for feature_name, points in face_landmarks.items():
                if feature_name != 'chin':  # Skip chin as it's already drawn
                    for i in range(len(points) - 1):
                        pt1 = points[i]
                        pt2 = points[i + 1]
                        cv2.line(processed_img, pt1, pt2, (0, 255, 0), 2)
                        # Draw triangle
                        if i < len(points) - 2:
                            pt3 = points[i + 2]
                            cv2.line(processed_img, pt2, pt3, (0, 255, 0), 2)
                            cv2.line(processed_img, pt3, pt1, (0, 255, 0), 2)

        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the extracted features and processed image
        return jsonify({
            "status": "success",
            "extracted": face_encodings[0].tolist(),
            "processed_image": f"data:image/jpeg;base64,{processed_image_base64}"
        })

    except Exception as e:
        print(f"Feature extraction error: {str(e)}")  # Debug log
        print(f"Error type: {type(e)}")  # Debug log
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Debug log
        return jsonify({"error": str(e)}), 500

@app.route("/api/face/optimize", methods=["POST"])
def optimize_features_endpoint():
    try:
        # Verify Firebase token
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "No token provided"}), 401
        
        decoded_token = verify_firebase_token(token)
        if not decoded_token:
            return jsonify({"error": "Invalid token"}), 401

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        image_data = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
        # Extract features
        start_time = time.time()
        extracted_features = extract_face_features(image_data)
        extraction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if extracted_features is None:
            return jsonify({"error": "No face detected in image"}), 400
            
        # Apply Whale Optimization Algorithm
        start_time = time.time()
        optimized_features = whale_optimization(extracted_features)
        optimization_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Calculate detailed metrics
        # 1. Storage Efficiency
        original_size = extracted_features.nbytes
        optimized_size = optimized_features.nbytes
        storage_reduction = ((original_size - optimized_size) / original_size) * 100

        # 2. Feature Distinctiveness
        original_variance = np.var(extracted_features)
        optimized_variance = np.var(optimized_features)
        distinctiveness_improvement = ((optimized_variance - original_variance) / original_variance) * 100

        # 3. Feature Robustness
        # Add small noise to test robustness
        noise = np.random.normal(0, 0.1, extracted_features.shape)
        noisy_original = extracted_features + noise
        noisy_optimized = optimized_features + noise
        
        # Calculate similarity with noisy versions
        original_robustness = np.dot(extracted_features, noisy_original) / (np.linalg.norm(extracted_features) * np.linalg.norm(noisy_original))
        optimized_robustness = np.dot(optimized_features, noisy_optimized) / (np.linalg.norm(optimized_features) * np.linalg.norm(noisy_optimized))
        robustness_improvement = ((optimized_robustness - original_robustness) / original_robustness) * 100

        # 4. Matching Speed
        # Simulate matching with 1000 random features
        test_features = np.random.rand(1000, len(extracted_features))
        start_time = time.time()
        for feature in test_features:
            np.dot(extracted_features, feature)
        original_matching_time = (time.time() - start_time) * 1000

        start_time = time.time()
        for feature in test_features:
            np.dot(optimized_features, feature)
        optimized_matching_time = (time.time() - start_time) * 1000
        matching_speed_improvement = ((original_matching_time - optimized_matching_time) / original_matching_time) * 100

        # Calculate overall accuracy
        accuracy = calculate_accuracy(extracted_features, optimized_features)

        return jsonify({
            "extracted": extracted_features.tolist(),
            "optimized": optimized_features.tolist(),
            "performance": {
                "extractionTime": extraction_time,
                "optimizationTime": optimization_time,
                "accuracy": accuracy,
                "storageEfficiency": {
                    "originalSize": original_size,
                    "optimizedSize": optimized_size,
                    "reductionPercentage": storage_reduction
                },
                "featureQuality": {
                    "originalVariance": original_variance,
                    "optimizedVariance": optimized_variance,
                    "distinctivenessImprovement": distinctiveness_improvement
                },
                "robustness": {
                    "originalRobustness": original_robustness,
                    "optimizedRobustness": optimized_robustness,
                    "improvementPercentage": robustness_improvement
                },
                "matchingSpeed": {
                    "originalMatchingTime": original_matching_time,
                    "optimizedMatchingTime": optimized_matching_time,
                    "improvementPercentage": matching_speed_improvement
                }
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/face/match", methods=["POST"])
def match_face():
    try:
        print("Starting face matching process...")  # Debug log
        
        # Verify Firebase token
        token = request.headers.get("Authorization")
        if not token:
            print("No token provided")  # Debug log
            return jsonify({"error": "No token provided"}), 401
        
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token.split(' ')[1]
        
        try:
            decoded_token = auth.verify_id_token(token)
            print(f"Decoded token: {decoded_token}")  # Debug log
        except Exception as e:
            print(f"Token verification error: {str(e)}")  # Debug log
            return jsonify({"error": "Invalid token"}), 401
        
        # Get image data
        if "image" not in request.files:
            print("No image provided in request")  # Debug log
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files["image"]
        print(f"Received image file: {image_file.filename}")  # Debug log
        image_data = image_file.read()
        
        # Convert image data to numpy array for face detection
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")  # Debug log
            return jsonify({"error": "Failed to decode image"}), 400
            
        print(f"Image shape: {img.shape}")  # Debug log
        
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check if face is detected
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            print("No face detected in image")  # Debug log
            return jsonify({"error": "No face detected in image. Please make sure your face is clearly visible."}), 400
            
        print(f"Found {len(face_locations)} faces in image")  # Debug log
        
        # Extract features from input image
        print("Extracting features from input image...")  # Debug log
        input_features = extract_face_features(image_data)
        if input_features is None:
            print("Failed to extract face features")  # Debug log
            return jsonify({"error": "Failed to extract face features. Please try again with a clearer image."}), 400
        
        print(f"Input features shape: {input_features.shape}")  # Debug log
        print(f"Input features range: [{np.min(input_features)}, {np.max(input_features)}]")  # Debug log
        
        # Get all users with registered faces
        print("Fetching registered users...")  # Debug log
        users_ref = db.collection("users").where("has_face_registered", "==", True).get()
        registered_users = list(users_ref)
        print(f"Found {len(registered_users)} registered users")  # Debug log
        
        if len(registered_users) == 0:
            print("No registered users found")  # Debug log
            return jsonify({"error": "No registered faces found in the system"}), 404
        
        best_match = None
        best_confidence = 0
        all_similarities = []
        all_user_ids = []
        
        # Process each registered user
        for user in registered_users:
            user_data = user.to_dict()
            try:
                print(f"Processing user: {user.id}")  # Debug log
                print(f"User data: {user_data}")  # Debug log
                
                # Check if face features exist
                if "face_features" not in user_data:
                    print(f"No face features found for user {user.id}")  # Debug log
                    continue
                
                try:
                    # Decode base64 and decrypt stored features
                    stored_features = decrypt_face_features(base64.b64decode(user_data["face_features"]))
                    
                    print(f"Successfully decrypted features for user {user.id}")  # Debug log
                    print(f"Stored features shape: {stored_features.shape}")  # Debug log
                    print(f"Stored features range: [{np.min(stored_features)}, {np.max(stored_features)}]")  # Debug log
                    
                    # Ensure features are in the correct format
                    if len(stored_features.shape) == 1:
                        stored_features = stored_features.reshape(1, -1)
                    if len(input_features.shape) == 1:
                        input_features = input_features.reshape(1, -1)
                    
                    # Calculate face distance directly
                    face_distance = face_recognition.face_distance(stored_features, input_features)[0]
                    similarity_score = 1 - face_distance  # Convert distance to similarity score
                    
                    print(f"Face distance for user {user.id}: {face_distance}")  # Debug log
                    print(f"Similarity score for user {user.id}: {similarity_score}")  # Debug log
                    
                    all_similarities.append(float(similarity_score))  # Convert to float
                    all_user_ids.append(str(user.id))  # Convert to string
                    
                    if similarity_score > best_confidence:
                        best_confidence = similarity_score
                        best_match = {
                            "user_id": str(user.id),  # Convert to string
                            "email": str(user_data.get("email", "")),  # Convert to string
                            "name": str(user_data.get("name", "")),  # Convert to string
                            "confidence": float(similarity_score)  # Convert to float
                        }
                except Exception as e:
                    print(f"Error processing features for user {user.id}: {str(e)}")  # Debug log
                    continue
                    
            except Exception as e:
                print(f"Error processing user {user.id}: {str(e)}")  # Debug log
                continue
        
        print(f"All similarity scores: {all_similarities}")  # Debug log
        print(f"All user IDs: {all_user_ids}")  # Debug log
        print(f"Best match confidence: {best_confidence}")  # Debug log
        
        # Use a more lenient threshold (0.5 instead of 0.6)
        MATCH_THRESHOLD = 0.5
        
        # Return detailed matching information with all values converted to JSON-serializable types
        response_data = {
            "match_found": bool(best_confidence >= MATCH_THRESHOLD),  # Convert to bool
            "best_match": best_match,
            "all_similarities": [float(x) for x in all_similarities],  # Convert to float
            "threshold": float(MATCH_THRESHOLD),  # Convert to float
            "debug_info": {
                "input_features_shape": list(input_features.shape),  # Convert to list
                "input_features_range": [float(np.min(input_features)), float(np.max(input_features))],
                "number_of_registered_users": int(len(registered_users)),  # Convert to int
                "best_confidence": float(best_confidence) if best_match else 0.0,  # Convert to float
                "threshold_used": float(MATCH_THRESHOLD),  # Convert to float
                "all_user_ids": [str(x) for x in all_user_ids],  # Convert to string
                "all_similarities": [float(x) for x in all_similarities]  # Convert to float
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Face matching error: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

# ==============================
# Whale Optimization Algorithm (WOA)
# ==============================
def whale_optimization(features, max_iter=50, n_whales=10):
    # Initialize parameters
    a = 2  # Initial value of a
    a_decrease = 2 / max_iter  # Linear decrease of a
    
    # Initialize whale positions with reduced random variation
    whales = np.zeros((n_whales, len(features)))
    for i in range(n_whales):
        # Start with small random variations around original features
        whales[i] = features * (1 + np.random.normal(0, 0.05, len(features)))  # Reduced variation
    
    best_position = features.copy()
    best_fitness = float('-inf')
    
    # Main optimization loop
    for iter in range(max_iter):
        a -= a_decrease  # Decrease a linearly
        
        for i in range(n_whales):
            # Update position
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            
            if abs(A) < 1:
                # Encircling prey
                D = abs(C * best_position - whales[i])
                whales[i] = best_position - A * D
            else:
                # Search for prey
                rand_whale = whales[np.random.randint(0, n_whales)]
                D = abs(C * rand_whale - whales[i])
                whales[i] = rand_whale - A * D
            
            # Ensure features stay within reasonable bounds
            whales[i] = np.clip(whales[i], 0, 1)
            
            # Calculate fitness
            fitness = calculate_fitness(features, whales[i])
            
            # Update best position
            if fitness > best_fitness:
                best_fitness = fitness
                best_position = whales[i].copy()
    
    return best_position

def calculate_fitness(original_features, candidate_features):
    # Calculate similarity score (cosine similarity)
    similarity = np.dot(original_features, candidate_features) / (
        np.linalg.norm(original_features) * np.linalg.norm(candidate_features)
    )
    
    # Calculate feature distinctiveness
    original_variance = np.var(original_features)
    candidate_variance = np.var(candidate_features)
    distinctiveness = min(candidate_variance / original_variance, 2.0)
    
    # Calculate dimensionality reduction score
    reduction = max(0, 1 - len(candidate_features) / len(original_features))
    
    # Combined fitness score with adjusted weights
    fitness = (
        0.6 * similarity +  # Higher weight on feature preservation
        0.25 * distinctiveness +  # Moderate weight on distinctiveness
        0.15 * reduction  # Lower weight on dimensionality reduction
    )
    
    return fitness

def calculate_accuracy(original_features, optimized_features):
    # Calculate similarity score (cosine similarity)
    similarity = np.dot(original_features, optimized_features) / (
        np.linalg.norm(original_features) * np.linalg.norm(optimized_features)
    )
    
    # Calculate feature distinctiveness (variance ratio)
    original_variance = np.var(original_features)
    optimized_variance = np.var(optimized_features)
    distinctiveness = min(optimized_variance / original_variance, 2.0)  # Cap at 2x improvement
    
    # Calculate dimensionality reduction score
    original_size = len(original_features)
    optimized_size = len(optimized_features)
    reduction_score = 1.0 if optimized_size < original_size else 0.8
    
    # Calculate robustness score (how well features handle noise)
    noise = np.random.normal(0, 0.1, len(original_features))
    noisy_original = original_features + noise
    noisy_optimized = optimized_features + noise
    robustness = np.dot(noisy_original, noisy_optimized) / (
        np.linalg.norm(noisy_original) * np.linalg.norm(noisy_optimized)
    )
    
    # Weighted accuracy calculation with adjusted weights
    accuracy = (
        0.5 * similarity +  # Feature preservation (50%)
        0.25 * distinctiveness +  # Feature distinctiveness (25%)
        0.15 * reduction_score +  # Dimensionality reduction (15%)
        0.1 * robustness  # Feature robustness (10%)
    )
    
    # Convert to percentage and ensure it's between 0 and 100
    accuracy_percentage = min(max(accuracy * 100, 0), 100)
    return accuracy_percentage

# ==============================
# Feature Extraction
# ==============================
def extract_face_features(image_data):
    try:
        print("Converting image data to numpy array...")  # Debug log
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image")  # Debug log
            return None
            
        print(f"Image shape: {img.shape}")  # Debug log
        
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract face encoding first
        print("Extracting face encodings...")  # Debug log
        face_encodings = face_recognition.face_encodings(rgb_image)
        if not face_encodings:
            print("No face encodings found")  # Debug log
            return None
            
        print(f"Face encoding shape: {face_encodings[0].shape}")  # Debug log
        return face_encodings[0]  # Return the first face encoding
        
    except Exception as e:
        print(f"Error in extract_face_features: {str(e)}")  # Debug log
        return None

# ==============================
# Encryption and Decryption
# ==============================
def encrypt_face_features(features):
    """Encrypt feature array using Fernet."""
    try:
        print("Encrypting features...")  # Debug log
        print(f"Input features shape: {features.shape}")  # Debug log
        print(f"Input features range: [{np.min(features)}, {np.max(features)}]")  # Debug log
        
        # Convert to list and serialize
        features_list = features.tolist()
        serialized = json.dumps(features_list)
        
        # Encrypt
        encrypted = cipher.encrypt(serialized.encode())
        
        # Convert to base64 string
        encrypted_base64 = base64.b64encode(encrypted).decode('utf-8')
        
        # Verify encryption
        try:
            # Decode base64 back to bytes
            encrypted_bytes = base64.b64decode(encrypted_base64)
            # Decrypt
            decrypted = cipher.decrypt(encrypted_bytes)
            decrypted_features = np.array(json.loads(decrypted.decode()))
            print(f"Verification successful - Decrypted features shape: {decrypted_features.shape}")  # Debug log
            print(f"Verification successful - Decrypted features range: [{np.min(decrypted_features)}, {np.max(decrypted_features)}]")  # Debug log
        except Exception as e:
            print(f"Verification error: {str(e)}")  # Debug log
            raise
        
        return encrypted_base64
    except Exception as e:
        print(f"Encryption error: {str(e)}")  # Debug log
        raise

def decrypt_face_features(encrypted_features):
    """Decrypt feature array."""
    try:
        print("Decrypting features...")  # Debug log
        
        # If input is a string (base64), decode it first
        if isinstance(encrypted_features, str):
            encrypted_features = base64.b64decode(encrypted_features)
        
        # Ensure proper padding for base64
        padding = 4 - (len(encrypted_features) % 4)
        if padding != 4:
            encrypted_features += b'=' * padding
            
        # Decrypt
        decrypted = cipher.decrypt(encrypted_features)
        features = np.array(json.loads(decrypted.decode()))
        
        print(f"Decrypted features shape: {features.shape}")  # Debug log
        print(f"Decrypted features range: [{np.min(features)}, {np.max(features)}]")  # Debug log
        
        return features
    except Exception as e:
        print(f"Decryption error: {str(e)}")  # Debug log
        print(f"Error type: {type(e)}")  # Debug log
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Debug log
        raise

# ==============================
# Classification
# ==============================
def classify_face(features):
    """Classify face using k-NN."""
    if len(database["features"]) == 0:
        return None
    decrypted_features = [decrypt_face_features(f) for f in database["features"]]
    labels = database["labels"]
    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn.fit(decrypted_features, labels)
    predicted_label = knn.predict([features])
    return predicted_label[0]

def verify_firebase_token(token):
    """Verify Firebase ID token."""
    try:
        # Remove 'Bearer ' prefix if present
        if token and isinstance(token, str):
            if token.startswith('Bearer '):
                token = token.split(' ')[1]
            
            # Verify the token
            decoded_token = auth.verify_id_token(token)
            print(f"Token verified successfully for user: {decoded_token['uid']}")  # Debug log
            return decoded_token
        else:
            print("Invalid token format")  # Debug log
            return None
    except Exception as e:
        print(f"Token verification error: {str(e)}")  # Debug log
        return None

def optimize_features(features):
    # Simple feature normalization
    return (features - np.mean(features)) / (np.std(features) + 1e-6)

@app.route("/api/security/encrypt", methods=["POST"])
def encrypt_features_endpoint():
    try:
        print("Starting encryption process...")  # Debug log
        
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            print("No authorization header found")  # Debug log
            return jsonify({"error": "No authorization token provided"}), 401

        # Verify the Firebase token
        decoded_token = verify_firebase_token(auth_header)
        if not decoded_token:
            print("Token verification failed")  # Debug log
            return jsonify({"error": "Invalid token"}), 401

        user_id = decoded_token['uid']
        print(f"Processing request for user: {user_id}")  # Debug log

        # Get features from request
        data = request.get_json()
        if not data:
            print("No JSON data in request")  # Debug log
            return jsonify({"error": "No data provided"}), 400
            
        if 'features' not in data:
            print("No features in request data")  # Debug log
            return jsonify({"error": "No features provided"}), 400

        print(f"Received features array of length: {len(data['features'])}")  # Debug log
        features = np.array(data['features'])
        print(f"Features shape: {features.shape}")  # Debug log
        
        # Encrypt the features
        print("Starting feature encryption...")  # Debug log
        encrypted_features = encrypt_face_features(features)
        print("Features encrypted successfully")  # Debug log
        
        # Return the encrypted features directly (it's already base64 encoded)
        return jsonify({
            "status": "success",
            "encrypted_features": encrypted_features
        })

    except Exception as e:
        print(f"Encryption error: {str(e)}")  # Debug log
        print(f"Error type: {type(e)}")  # Debug log
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Debug log
        return jsonify({"error": str(e)}), 500

@app.route("/api/security/decrypt", methods=["POST"])
def decrypt_features_endpoint():
    try:
        print("Starting decryption process...")  # Debug log
        
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            print("No authorization header found")  # Debug log
            return jsonify({"error": "No authorization token provided"}), 401

        # Verify the Firebase token
        decoded_token = verify_firebase_token(auth_header)
        if not decoded_token:
            print("Token verification failed")  # Debug log
            return jsonify({"error": "Invalid token"}), 401

        user_id = decoded_token['uid']
        print(f"Processing request for user: {user_id}")  # Debug log

        # Get encrypted features from request
        data = request.get_json()
        if not data:
            print("No JSON data in request")  # Debug log
            return jsonify({"error": "No data provided"}), 400
            
        if 'features' not in data:
            print("No features in request data")  # Debug log
            return jsonify({"error": "No encrypted features provided"}), 400

        print(f"Received encrypted features of length: {len(data['features'])}")  # Debug log
        
        # Convert base64 back to bytes
        try:
            encrypted_features = base64.b64decode(data['features'])
            print("Successfully decoded base64 features")  # Debug log
        except Exception as e:
            print(f"Base64 decoding error: {str(e)}")  # Debug log
            return jsonify({"error": "Invalid base64 encoding"}), 400
        
        # Decrypt the features
        print("Starting feature decryption...")  # Debug log
        decrypted_features = decrypt_face_features(encrypted_features)
        print("Features decrypted successfully")  # Debug log
        print(f"Decrypted features shape: {decrypted_features.shape}")  # Debug log
        
        return jsonify({
            "status": "success",
            "decrypted_features": decrypted_features.tolist()
        })

    except Exception as e:
        print(f"Decryption error: {str(e)}")  # Debug log
        print(f"Error type: {type(e)}")  # Debug log
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Debug log
        return jsonify({"error": str(e)}), 500

# ==============================
# Run Flask Server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)