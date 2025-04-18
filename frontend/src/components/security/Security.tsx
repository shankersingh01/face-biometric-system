import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Fade,
} from "@mui/material";
import {
  CloudUpload as CloudUploadIcon,
  Person as PersonIcon,
} from "@mui/icons-material";
import api from "../../utils/axiosConfig";
import { auth } from "../../config/firebase";
import { useNavigate } from "react-router-dom";
import {
  getLastRegisteredImage,
  clearLastRegisteredImage,
} from "../../utils/imageUtils";

const Security: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | undefined>(undefined);
  const [lastRegisteredImage, setLastRegisteredImage] = useState<string | null>(
    null
  );
  const [processedImageUrl, setProcessedImageUrl] = useState<
    string | undefined
  >(undefined);
  const [features, setFeatures] = useState<{
    original: number[];
    encrypted: string;
    decrypted: number[];
  }>({
    original: [],
    encrypted: "",
    decrypted: [],
  });

  // Check authentication state
  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (!user) {
        navigate("/login");
      }
    });

    return () => unsubscribe();
  }, [navigate]);

  // Load last registered image
  useEffect(() => {
    const loadLastRegisteredImage = async () => {
      try {
        console.log("Loading last registered image...");
        const imageUrl = await getLastRegisteredImage();
        console.log("Last registered image URL:", imageUrl);
        if (imageUrl) {
          setLastRegisteredImage(imageUrl);
        }
      } catch (error) {
        console.error("Error loading last registered image:", error);
      }
    };

    loadLastRegisteredImage();

    return () => {
      if (lastRegisteredImage) {
        clearLastRegisteredImage();
      }
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, []);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setProcessedImageUrl(undefined);
      setError(null);
      setFeatures({
        original: [],
        encrypted: "",
        decrypted: [],
      });
    }
  };

  const handleUseLastRegisteredImage = async () => {
    if (lastRegisteredImage) {
      try {
        // Fetch the last registered image as a blob
        const response = await fetch(lastRegisteredImage);
        const blob = await response.blob();
        const file = new File([blob], "last_registered.jpg", {
          type: "image/jpeg",
        });

        // Update both states
        setSelectedImage(file);
        setPreviewUrl(lastRegisteredImage);
        setProcessedImageUrl(undefined);
        setError(null);
        setFeatures({
          original: [],
          encrypted: "",
          decrypted: [],
        });
      } catch (error) {
        console.error("Error using last registered image:", error);
        setError("Failed to load last registered image");
      }
    }
  };

  const handleProcessFeatures = async () => {
    if (!previewUrl) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    const formData = new FormData();

    try {
      // If using last registered image, fetch it as a blob
      if (previewUrl.startsWith("blob:")) {
        const response = await fetch(previewUrl);
        const blob = await response.blob();
        formData.append("image", blob, "last_registered.jpg");
      } else if (selectedImage) {
        formData.append("image", selectedImage);
      } else {
        throw new Error("No image selected for processing");
      }

      // First extract features
      const extractResponse = await api.post("/api/face/extract", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (!extractResponse.data || !extractResponse.data.extracted) {
        throw new Error("Failed to extract features from image");
      }

      const originalFeatures = extractResponse.data.extracted;
      setProcessedImageUrl(extractResponse.data.processed_image);

      // Then encrypt the features
      const encryptResponse = await api.post("/api/security/encrypt", {
        features: originalFeatures,
      });

      if (!encryptResponse.data || !encryptResponse.data.encrypted_features) {
        throw new Error("Failed to encrypt features");
      }

      // Finally decrypt the features
      const decryptResponse = await api.post("/api/security/decrypt", {
        features: encryptResponse.data.encrypted_features,
      });

      if (!decryptResponse.data || !decryptResponse.data.decrypted_features) {
        throw new Error("Failed to decrypt features");
      }

      setFeatures({
        original: originalFeatures,
        encrypted: encryptResponse.data.encrypted_features,
        decrypted: decryptResponse.data.decrypted_features,
      });

      setSuccess("Features processed successfully!");
    } catch (err: any) {
      console.error("Error processing features:", err);
      setError(err.message || "Failed to process features");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Encryption/Decryption
        </Typography>
        <Grid container spacing={3}>
          {/* Last Registered Image */}
          <Grid item xs={12} md={4}>
            {lastRegisteredImage && (
              <Fade in={true} timeout={500}>
                <Box
                  sx={{
                    p: 2,
                    border: "2px dashed",
                    borderColor: "primary.main",
                    borderRadius: 2,
                    backgroundColor: "background.paper",
                    transition: "all 0.3s ease",
                    "&:hover": {
                      backgroundColor: "action.hover",
                      transform: "translateY(-2px)",
                    },
                  }}
                >
                  <Typography
                    variant="subtitle1"
                    gutterBottom
                    sx={{
                      color: "text.secondary",
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                    }}
                  >
                    <PersonIcon color="primary" />
                    Last Registered Image
                  </Typography>
                  <Box
                    component="img"
                    src={lastRegisteredImage}
                    alt="Last Registered"
                    sx={{
                      width: "100%",
                      maxHeight: 200,
                      objectFit: "contain",
                      mb: 2,
                      borderRadius: 2,
                      boxShadow: 3,
                      transition: "all 0.3s ease",
                      "&:hover": {
                        transform: "scale(1.02)",
                      },
                    }}
                  />
                  <Button
                    variant="contained"
                    onClick={handleUseLastRegisteredImage}
                    fullWidth
                    sx={{
                      height: 48,
                      fontSize: "1rem",
                      fontWeight: "bold",
                      borderRadius: 2,
                    }}
                  >
                    Use This Image
                  </Button>
                </Box>
              </Fade>
            )}
          </Grid>

          {/* Original Image Upload */}
          <Grid item xs={12} md={4}>
            <Box
              sx={{
                border: "2px dashed #ccc",
                borderRadius: 2,
                p: 2,
                textAlign: "center",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              {previewUrl ? (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Original Image
                  </Typography>
                  <img
                    src={previewUrl}
                    alt="Selected"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "300px",
                      objectFit: "contain",
                    }}
                  />
                </Box>
              ) : (
                <>
                  <CloudUploadIcon
                    sx={{ fontSize: 48, color: "primary.main" }}
                  />
                  <Typography variant="h6" gutterBottom>
                    Upload Image
                  </Typography>
                  <input
                    accept="image/*"
                    style={{ display: "none" }}
                    id="image-upload"
                    type="file"
                    onChange={handleImageSelect}
                  />
                  <label htmlFor="image-upload">
                    <Button variant="contained" component="span">
                      Select Image
                    </Button>
                  </label>
                </>
              )}
            </Box>
          </Grid>

          {/* Processed Image */}
          <Grid item xs={12} md={4}>
            <Box
              sx={{
                border: "2px dashed #ccc",
                borderRadius: 2,
                p: 2,
                textAlign: "center",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
              }}
            >
              {processedImageUrl ? (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Processed Image
                  </Typography>
                  <img
                    src={processedImageUrl}
                    alt="Processed"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "300px",
                      objectFit: "contain",
                    }}
                  />
                </Box>
              ) : (
                <Typography variant="body1" color="text.secondary">
                  Processed image will appear here
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>
        <Box sx={{ mt: 3, textAlign: "center" }}>
          <Button
            variant="contained"
            onClick={handleProcessFeatures}
            disabled={!previewUrl || loading}
            sx={{ mr: 2 }}
          >
            {loading ? <CircularProgress size={24} /> : "Process Features"}
          </Button>
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          {success && (
            <Alert severity="success" sx={{ mt: 2 }}>
              {success}
            </Alert>
          )}
        </Box>
        {features.original.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Feature Comparison
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Original Features
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 2 }}
                    >
                      Face template before encryption
                    </Typography>
                    <Box
                      sx={{
                        maxHeight: "300px",
                        overflow: "auto",
                        fontFamily: "monospace",
                        fontSize: "0.8rem",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {JSON.stringify(features.original, null, 2)}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Encrypted Format
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 2 }}
                    >
                      As stored in Firebase
                    </Typography>
                    <Box
                      sx={{
                        maxHeight: "300px",
                        overflow: "auto",
                        fontFamily: "monospace",
                        fontSize: "0.8rem",
                        wordBreak: "break-all",
                      }}
                    >
                      {features.encrypted}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Decrypted Features
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 2 }}
                    >
                      After decryption
                    </Typography>
                    <Box
                      sx={{
                        maxHeight: "300px",
                        overflow: "auto",
                        fontFamily: "monospace",
                        fontSize: "0.8rem",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {JSON.stringify(features.decrypted, null, 2)}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default Security;
