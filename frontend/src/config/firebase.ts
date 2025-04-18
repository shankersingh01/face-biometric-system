import { initializeApp, FirebaseApp } from 'firebase/app';
import { getAuth, Auth } from 'firebase/auth';
import { getFirestore, Firestore } from 'firebase/firestore';

const firebaseConfig = {
    apiKey: "AIzaSyA4aJ70ZPwBI0iDPS1whamFsEZ2bheOcgk",
    authDomain: "face-biometric-system.firebaseapp.com",
    projectId: "face-biometric-system",
    storageBucket: "face-biometric-system.appspot.com",
    messagingSenderId: "437997691196",
    appId: "1:437997691196:web:ce66c126fcad6e46f1be39",
    measurementId: "G-PKL14S9GFM"
  };

// Initialize Firebase
const app: FirebaseApp = initializeApp(firebaseConfig);
export const auth: Auth = getAuth(app);
export const db: Firestore = getFirestore(app); 