export interface User {
  id: string;
  email: string;
  name: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface FaceFeatures {
  original: number[];
  optimized: number[];
  encrypted: string;
}

export interface RecognitionResult {
  match: boolean;
  confidence: number;
  user?: User;
}

export interface OptimizationComparison {
  originalTime: number;
  optimizedTime: number;
  originalAccuracy: number;
  optimizedAccuracy: number;
} 