import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { ThemeProvider, createTheme } from "@mui/material";
import CssBaseline from "@mui/material/CssBaseline";
import { AuthProvider, useAuth } from "./contexts/AuthContext";

// Components
import Login from "./components/auth/Login";
import Signup from "./components/auth/Signup";
import DashboardLayout from "./components/dashboard/DashboardLayout";
import Dashboard from "./components/dashboard/Dashboard";
import FaceRegistration from "./components/face-recognition/FaceRegistration";
import FeatureExtraction from "./components/face-recognition/FeatureExtraction";
import WhaleOptimization from "./components/face-recognition/WhaleOptimization";
import FaceMatching from "./components/face-recognition/FaceMatching";
import Security from "./components/security/Security";
import EncryptionDecryption from "./components/face-recognition/EncryptionDecryption";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#dc004e",
    },
  },
});

const PrivateRoute: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { currentUser } = useAuth();
  return currentUser ? <>{children}</> : <Navigate to="/login" />;
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route
              path="/dashboard"
              element={
                <PrivateRoute>
                  <DashboardLayout />
                </PrivateRoute>
              }
            >
              <Route index element={<Dashboard />} />
              <Route path="register" element={<FaceRegistration />} />
              <Route path="features" element={<FeatureExtraction />} />
              <Route path="optimization" element={<WhaleOptimization />} />
              <Route path="security" element={<Security />} />
              <Route path="matching" element={<FaceMatching />} />
            </Route>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Router>
      </ThemeProvider>
    </AuthProvider>
  );
};

export default App;
