import { auth } from "../config/firebase";
import { onAuthStateChanged, User } from "firebase/auth";

let currentUser: User | null = null;

// Listen for auth state changes
onAuthStateChanged(auth, (user) => {
  currentUser = user;
});

export const getAuthToken = async (): Promise<string | null> => {
  try {
    if (!currentUser) {
      return null;
    }

    // Get a fresh token
    const token = await currentUser.getIdToken(true);
    
    // Store the token in localStorage
    localStorage.setItem("token", token);
    
    return token;
  } catch (error) {
    console.error("Error getting auth token:", error);
    // Don't clear the token on error, let the caller handle it
    return localStorage.getItem("token");
  }
};

export const isTokenExpired = (token: string): boolean => {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    // Add a 5-minute buffer to prevent edge cases
    return payload.exp * 1000 < Date.now() + 5 * 60 * 1000;
  } catch (error) {
    console.error("Error checking token expiration:", error);
    return true;
  }
}; 