# Face Biometric System

A secure face recognition system that uses facial features for user authentication and identification. The system includes feature extraction, encryption, and matching capabilities.

## Features

- Face registration and recognition
- Secure feature storage with encryption
- Real-time face matching
- User authentication with Firebase
- Feature optimization using Whale Optimization Algorithm
- Facial landmark detection and visualization

## Tech Stack

- Backend: Python (Flask)
- Frontend: React with TypeScript
- Authentication: Firebase
- Face Recognition: face_recognition library
- Database: Firebase Firestore
- Security: AES encryption for feature storage

## Prerequisites

- Python 3.8+
- Node.js 14+
- Firebase account and project
- Firebase service account key

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-biometric-system.git
cd face-biometric-system
```

2. Set up the backend:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Firebase credentials
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

4. Add your Firebase service account key:
- Place your `firebase-service-account.json` in the root directory

## Running the Application

1. Start the backend server:
```bash
# From the root directory
python app.py
```

2. Start the frontend development server:
```bash
# From the frontend directory
npm start
```

3. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## API Endpoints

- `/api/auth/signup` - User registration
- `/api/auth/login` - User login
- `/api/face/register` - Face registration
- `/api/face/match` - Face matching
- `/api/face/extract` - Feature extraction
- `/api/face/optimize` - Feature optimization
- `/api/security/encrypt` - Feature encryption
- `/api/security/decrypt` - Feature decryption

## Security Features

- Face feature encryption before storage
- Secure token-based authentication
- Protected API endpoints
- Feature optimization for better matching

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [Firebase](https://firebase.google.com/) for authentication and database
- [Flask](https://flask.palletsprojects.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework 