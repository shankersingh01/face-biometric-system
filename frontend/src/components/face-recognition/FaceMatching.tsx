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
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
} from "@mui/icons-material";
import axios from "axios";
import { getAuthToken } from "../../utils/auth";
import {
  getLastRegisteredImage,
  clearLastRegisteredImage,
} from "../../utils/imageUtils";

const FaceMatching: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [lastRegisteredImage, setLastRegisteredImage] = useState<string | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matchingData, setMatchingData] = useState<{
    match_found: boolean;
    best_match?: {
      user_id: string;
      name: string;
      email: string;
      confidence: number;
    };
    debug_info?: {
      best_confidence: number;
    };
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
      setMatchingData(null);
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
      setMatchingData(null);
    }
  };

  const handleMatch = async () => {
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
        "http://127.0.0.1:5000/api/face/match",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setMatchingData(response.data);
    } catch (error) {
      console.error("Matching error:", error);
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
        Face Matching
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

        {/* Match Button */}
        <Button
          variant="contained"
          onClick={handleMatch}
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
          {loading ? <CircularProgress size={24} /> : "Match Face"}
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
          Matching Results
        </Typography>

        {matchingData ? (
          <Fade in={true} timeout={500}>
            <Box sx={{ mt: 2 }}>
              <Card
                sx={{
                  borderRadius: 2,
                  overflow: "hidden",
                  backgroundColor: matchingData.match_found
                    ? "success.light"
                    : "error.light",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    transform: "translateY(-2px)",
                    boxShadow: 6,
                  },
                }}
              >
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ textAlign: "center", mb: 3 }}>
                    {matchingData.match_found ? (
                      <Zoom in={true} timeout={500}>
                        <CheckCircleIcon
                          sx={{
                            fontSize: 64,
                            color: "success.dark",
                            mb: 2,
                          }}
                        />
                      </Zoom>
                    ) : (
                      <Zoom in={true} timeout={500}>
                        <CancelIcon
                          sx={{
                            fontSize: 64,
                            color: "error.dark",
                            mb: 2,
                          }}
                        />
                      </Zoom>
                    )}
                    <Typography
                      variant="h4"
                      sx={{
                        color: matchingData.match_found
                          ? "success.dark"
                          : "error.dark",
                        fontWeight: "bold",
                        mb: 2,
                      }}
                    >
                      {matchingData.match_found ? "MATCH FOUND" : "NO MATCH"}
                    </Typography>
                  </Box>

                  {matchingData.match_found && matchingData.best_match && (
                    <Box sx={{ mt: 2 }}>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="h6" sx={{ mb: 1 }}>
                        User Details
                      </Typography>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body1" sx={{ mb: 1 }}>
                          <strong>Name:</strong> {matchingData.best_match.name}
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 1 }}>
                          <strong>Email:</strong>{" "}
                          {matchingData.best_match.email}
                        </Typography>
                      </Box>
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="body1" gutterBottom>
                          Confidence Level
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={matchingData.best_match.confidence * 100}
                          sx={{
                            height: 10,
                            borderRadius: 5,
                            backgroundColor: "rgba(255, 255, 255, 0.3)",
                            "& .MuiLinearProgress-bar": {
                              backgroundColor: "success.dark",
                            },
                          }}
                        />
                        <Typography
                          variant="body1"
                          sx={{
                            mt: 1,
                            textAlign: "right",
                            color: "success.dark",
                            fontWeight: "bold",
                          }}
                        >
                          {(matchingData.best_match.confidence * 100).toFixed(
                            2
                          )}
                          %
                        </Typography>
                      </Box>
                    </Box>
                  )}

                  {!matchingData.match_found && matchingData.debug_info && (
                    <Box sx={{ mt: 2 }}>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="body1" gutterBottom>
                        Best Confidence Level
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={matchingData.debug_info.best_confidence * 100}
                        sx={{
                          height: 10,
                          borderRadius: 5,
                          backgroundColor: "rgba(255, 255, 255, 0.3)",
                          "& .MuiLinearProgress-bar": {
                            backgroundColor: "error.dark",
                          },
                        }}
                      />
                      <Typography
                        variant="body1"
                        sx={{
                          mt: 1,
                          textAlign: "right",
                          color: "error.dark",
                          fontWeight: "bold",
                        }}
                      >
                        {(
                          matchingData.debug_info.best_confidence * 100
                        ).toFixed(2)}
                        %
                      </Typography>
                    </Box>
                  )}
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
              Select an image and click "Match Face" to see results
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default FaceMatching;
