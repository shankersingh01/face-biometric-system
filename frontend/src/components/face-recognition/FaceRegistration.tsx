import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Container,
  Typography,
  Paper,
  Alert,
  CircularProgress,
} from "@mui/material";
import { CloudUpload } from "@mui/icons-material";
import axios from "axios";
import { getAuthToken } from "../../utils/auth";
import {
  setLastRegisteredImage,
  clearLastRegisteredImage,
} from "../../utils/imageUtils";

const FaceRegistration: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");

  useEffect(() => {
    return () => {
      // Clean up the preview URL when component unmounts
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const previewUrl = URL.createObjectURL(file);
      setPreview(previewUrl);
      setError("");
      setSuccess("");
    }
  };

  const handleRegister = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const token = await getAuthToken();
      const formData = new FormData();
      formData.append("image", selectedImage);

      console.log("Sending request to register face..."); // Debug log
      const response = await axios.post(
        "http://127.0.0.1:5000/api/face/register",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
            "X-Requested-With": "XMLHttpRequest",
          },
          withCredentials: true,
        }
      );

      console.log("Response:", response.data); // Debug log
      setSuccess("Face registered successfully!");

      // Convert the image to base64 and store it
      const imageUrl = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          if (typeof reader.result === "string") {
            resolve(reader.result);
          }
        };
        reader.readAsDataURL(selectedImage);
      });

      console.log("Storing image in localStorage...");
      setLastRegisteredImage(imageUrl);
      console.log("Image stored successfully");

      setSelectedImage(null);
      setPreview("");
    } catch (err: any) {
      console.error("Error details:", err); // Debug log
      if (err.request) {
        setError(
          "Cannot connect to the server. Please make sure the backend server is running at http://127.0.0.1:5000"
        );
      } else {
        setError(
          err.response?.data?.error ||
            "Failed to register face. Please try again."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h5" gutterBottom>
          Register Your Face
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Upload a clear photo of your face for registration. Make sure your
          face is clearly visible and well-lit.
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}

        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 2,
          }}
        >
          <input
            accept="image/*"
            style={{ display: "none" }}
            id="face-image-upload"
            type="file"
            onChange={handleImageSelect}
          />
          <label htmlFor="face-image-upload">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUpload />}
              disabled={loading}
            >
              Select Image
            </Button>
          </label>

          {preview && (
            <Box
              component="img"
              src={preview}
              alt="Preview"
              sx={{
                maxWidth: "100%",
                maxHeight: 300,
                objectFit: "contain",
                mt: 2,
              }}
            />
          )}

          <Button
            variant="contained"
            onClick={handleRegister}
            disabled={!selectedImage || loading}
            sx={{ mt: 2 }}
          >
            {loading ? <CircularProgress size={24} /> : "Register Face"}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default FaceRegistration;
