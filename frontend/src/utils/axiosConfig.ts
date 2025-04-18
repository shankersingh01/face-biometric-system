import axios from "axios";
import { getAuthToken } from "./tokenManager";

const api = axios.create({
  baseURL: "http://127.0.0.1:5000",
  headers: {
    "Content-Type": "application/json",
  },
});

let isRefreshing = false;
let failedQueue: any[] = [];

const processQueue = (error: any = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve();
    }
  });
  failedQueue = [];
};

// Request interceptor
api.interceptors.request.use(
  async (config) => {
    try {
      const token = await getAuthToken();
      if (token) {
        // Ensure token is properly formatted with 'Bearer ' prefix
        const formattedToken = token.startsWith("Bearer ") ? token : `Bearer ${token}`;
        config.headers.Authorization = formattedToken;
      }
    } catch (error) {
      console.error("Error getting auth token:", error);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If the error is not 401 or the request has already been retried, reject
    if (error.response?.status !== 401 || originalRequest._retry) {
      return Promise.reject(error);
    }

    if (isRefreshing) {
      // If token refresh is in progress, queue the request
      return new Promise((resolve, reject) => {
        failedQueue.push({ resolve, reject });
      })
        .then(() => {
          return api(originalRequest);
        })
        .catch((err) => {
          return Promise.reject(err);
        });
    }

    originalRequest._retry = true;
    isRefreshing = true;

    try {
      const newToken = await getAuthToken();
      if (!newToken) {
        processQueue(new Error("Failed to refresh token"));
        window.location.href = "/login";
        return Promise.reject(error);
      }

      // Update the token in the original request
      const formattedToken = newToken.startsWith("Bearer ") ? newToken : `Bearer ${newToken}`;
      originalRequest.headers.Authorization = formattedToken;

      processQueue();
      return api(originalRequest);
    } catch (refreshError) {
      processQueue(refreshError);
      window.location.href = "/login";
      return Promise.reject(refreshError);
    } finally {
      isRefreshing = false;
    }
  }
);

export default api; 