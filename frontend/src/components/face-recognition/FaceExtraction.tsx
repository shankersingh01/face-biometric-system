import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  LinearProgress,
  Fade,
  Zoom,
  Card,
  CardContent,
  Divider,
} from "@mui/material";
import {
  CloudUpload as CloudUploadIcon,
  Person as PersonIcon,
  Fingerprint as FingerprintIcon,
} from "@mui/icons-material";
import axios from "axios";
import { getAuthToken } from "../../utils/auth";
import {
  getLastRegisteredImage,
  clearLastRegisteredImage,
} from "../../utils/imageUtils";

const FaceExtraction: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [lastRegisteredImage, setLastRegisteredImage] = useState<string | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extractionData, setExtractionData] = useState<{
    features: number[];
    processing_time: number;
    feature_quality: number;
  } | null>(null);

  useEffect(() => {
    const loadLastRegisteredImage = async () => {
      try {
        const imageUrl = await getLastRegisteredImage();
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
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage);
      }
    };
  }, []);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Clean up previous selected image if exists
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage);
      }
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      setError(null);
      setExtractionData(null);
    }
  };

  const handleUseLastRegisteredImage = () => {
    if (lastRegisteredImage) {
      // Clean up previous selected image if exists
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage);
      }
      setSelectedImage(lastRegisteredImage);
      setError(null);
      setExtractionData(null);
    }
  };

  const handleExtract = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const token = await getAuthToken();
      const formData = new FormData();

      // If the selected image is a URL (from last registered), fetch it as a blob
      if (selectedImage.startsWith("blob:")) {
        const response = await fetch(selectedImage);
        const blob = await response.blob();
        formData.append("image", blob, "last_registered.jpg");
      } else {
        // If it's a file upload, use it directly
        const response = await fetch(selectedImage);
        const blob = await response.blob();
        formData.append("image", blob, "uploaded.jpg");
      }

      const response = await axios.post(
        "http://127.0.0.1:5000/api/face/extract",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setExtractionData(response.data);
    } catch (error) {
      console.error("Extraction error:", error);
      if (axios.isAxiosError(error)) {
        if (error.response) {
          setError(`Server error: ${error.response.data.error}`);
        } else if (error.request) {
          setError("Network error: Could not connect to the server");
        } else {
          setError(`Error: ${error.message}`);
        }
      } else {
        setError("An unexpected error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography
        variant="h4"
        gutterBottom
        sx={{
          mb: 4,
          color: "primary.main",
          fontWeight: "bold",
          textAlign: "center",
        }}
      >
        Feature Extraction
      </Typography>

      {/* Image Selection Section */}
      <Paper
        elevation={3}
        sx={{
          p: 3,
          mb: 4,
          borderRadius: 2,
          transition: "all 0.3s ease",
          "&:hover": {
            boxShadow: 6,
          },
        }}
      >
        <Typography
          variant="h6"
          gutterBottom
          sx={{
            color: "primary.main",
            fontWeight: "bold",
            mb: 3,
          }}
        >
          Image Selection
        </Typography>

        <Grid container spacing={3}>
          {/* Last Registered Image */}
          <Grid item xs={12} md={6}>
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

          {/* Upload New Image */}
          <Grid item xs={12} md={6}>
            <Box
              sx={{
                border: "2px dashed",
                borderColor: "primary.main",
                borderRadius: 2,
                p: 3,
                textAlign: "center",
                cursor: "pointer",
                backgroundColor: "background.paper",
                transition: "all 0.3s ease",
                "&:hover": {
                  backgroundColor: "action.hover",
                  transform: "translateY(-2px)",
                },
              }}
            >
              <input
                accept="image/*"
                style={{ display: "none" }}
                id="image-upload"
                type="file"
                onChange={handleImageSelect}
              />
              <label htmlFor="image-upload">
                <Button
                  component="span"
                  variant="outlined"
                  startIcon={<CloudUploadIcon />}
                  sx={{
                    height: 48,
                    fontSize: "1rem",
                    fontWeight: "bold",
                    borderRadius: 2,
                    mb: 2,
                  }}
                >
                  Upload New Image
                </Button>
              </label>
              <Typography variant="body2" color="text.secondary">
                or drag and drop an image here
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Selected Image Preview */}
        {selectedImage && (
          <Fade in={true} timeout={500}>
            <Box sx={{ mt: 3 }}>
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
                Selected Image
              </Typography>
              <Box
                component="img"
                src={selectedImage}
                alt="Selected"
                sx={{
                  width: "100%",
                  maxHeight: 300,
                  objectFit: "contain",
                  borderRadius: 2,
                  boxShadow: 3,
                  transition: "all 0.3s ease",
                  "&:hover": {
                    transform: "scale(1.02)",
                  },
                }}
              />
            </Box>
          </Fade>
        )}

        {/* Error Message */}
        {error && (
          <Fade in={true} timeout={500}>
            <Alert
              severity="error"
              sx={{
                mt: 3,
                borderRadius: 2,
              }}
            >
              {error}
            </Alert>
          </Fade>
        )}

        {/* Extract Button */}
        <Button
          variant="contained"
          onClick={handleExtract}
          disabled={!selectedImage || loading}
          fullWidth
          size="large"
          sx={{
            mt: 3,
            height: 56,
            fontSize: "1.1rem",
            fontWeight: "bold",
            borderRadius: 2,
            transition: "all 0.3s ease",
            "&:hover": {
              transform: "translateY(-2px)",
              boxShadow: 6,
            },
          }}
        >
          {loading ? <CircularProgress size={24} /> : "Extract Features"}
        </Button>
      </Paper>

      {/* Results Section */}
      <Paper
        elevation={3}
        sx={{
          p: 3,
          borderRadius: 2,
          transition: "all 0.3s ease",
          "&:hover": {
            boxShadow: 6,
          },
        }}
      >
        <Typography
          variant="h6"
          gutterBottom
          sx={{
            color: "primary.main",
            fontWeight: "bold",
            mb: 3,
          }}
        >
          Extraction Results
        </Typography>

        {extractionData ? (
          <Fade in={true} timeout={500}>
            <Box sx={{ mt: 2 }}>
              <Card
                sx={{
                  borderRadius: 2,
                  overflow: "hidden",
                  backgroundColor: "background.paper",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    transform: "translateY(-2px)",
                    boxShadow: 6,
                  },
                }}
              >
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ textAlign: "center", mb: 3 }}>
                    <Zoom in={true} timeout={500}>
                      <FingerprintIcon
                        sx={{
                          fontSize: 64,
                          color: "primary.main",
                          mb: 2,
                        }}
                      />
                    </Zoom>
                    <Typography
                      variant="h4"
                      sx={{
                        color: "primary.main",
                        fontWeight: "bold",
                        mb: 2,
                      }}
                    >
                      FEATURES EXTRACTED
                    </Typography>
                  </Box>

                  <Box sx={{ mt: 2 }}>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      Processing Metrics
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body1" sx={{ mb: 1 }}>
                        <strong>Processing Time:</strong>{" "}
                        {extractionData.processing_time.toFixed(2)}ms
                      </Typography>
                      <Typography variant="body1" sx={{ mb: 1 }}>
                        <strong>Feature Quality:</strong>
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={extractionData.feature_quality * 100}
                        sx={{
                          height: 10,
                          borderRadius: 5,
                          backgroundColor: "rgba(0, 0, 0, 0.1)",
                          "& .MuiLinearProgress-bar": {
                            backgroundColor: "primary.main",
                          },
                        }}
                      />
                      <Typography
                        variant="body1"
                        sx={{
                          mt: 1,
                          textAlign: "right",
                          color: "primary.main",
                          fontWeight: "bold",
                        }}
                      >
                        {(extractionData.feature_quality * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ mt: 2 }}>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      Extracted Features
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {extractionData.features.length} features extracted
                    </Typography>
                    <Box
                      sx={{
                        mt: 2,
                        p: 2,
                        bgcolor: "background.default",
                        borderRadius: 1,
                        maxHeight: 200,
                        overflow: "auto",
                      }}
                    >
                      <Typography variant="body2" component="pre">
                        {JSON.stringify(extractionData.features, null, 2)}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Box>
          </Fade>
        ) : (
          <Box
            sx={{
              height: 200,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              bgcolor: "background.default",
              borderRadius: 2,
              transition: "all 0.3s ease",
              "&:hover": {
                backgroundColor: "action.hover",
              },
            }}
          >
            <Typography color="text.secondary" sx={{ textAlign: "center" }}>
              Select an image and click "Extract Features" to see results
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default FaceExtraction;
