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
  TextField,
} from "@mui/material";
import {
  CloudUpload as CloudUploadIcon,
  Person as PersonIcon,
  Lock as LockIcon,
  LockOpen as LockOpenIcon,
  Security as SecurityIcon,
} from "@mui/icons-material";
import axios from "axios";
import { getAuthToken } from "../../utils/auth";
import {
  getLastRegisteredImage,
  clearLastRegisteredImage,
} from "../../utils/imageUtils";

const EncryptionDecryption: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [lastRegisteredImage, setLastRegisteredImage] = useState<string | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [key, setKey] = useState("");
  const [operation, setOperation] = useState<"encrypt" | "decrypt">("encrypt");
  const [result, setResult] = useState<{
    processed_image: string;
    processing_time: number;
    security_level: number;
  } | null>(null);

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
      setResult(null);
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
      setResult(null);
    }
  };

  const handleProcess = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }

    if (!key) {
      setError("Please enter a key");
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

      formData.append("key", key);
      formData.append("operation", operation);

      const response = await axios.post(
        "http://127.0.0.1:5000/api/face/process",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (error) {
      console.error("Processing error:", error);
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
        {operation === "encrypt" ? "Encryption" : "Decryption"}
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
                <CloudUploadIcon color="primary" />
                Upload New Image
              </Typography>
              <input
                accept="image/*"
                style={{ display: "none" }}
                id="image-upload"
                type="file"
                onChange={handleImageSelect}
              />
              <label htmlFor="image-upload">
                <Button
                  variant="outlined"
                  component="span"
                  fullWidth
                  sx={{
                    height: 48,
                    fontSize: "1rem",
                    fontWeight: "bold",
                    borderRadius: 2,
                    mb: 2,
                  }}
                >
                  Choose File
                </Button>
              </label>
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

        {/* Key Input */}
        <Box sx={{ mt: 3 }}>
          <TextField
            fullWidth
            label="Encryption/Decryption Key"
            variant="outlined"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            type="password"
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: 2,
                "&:hover .MuiOutlinedInput-notchedOutline": {
                  borderColor: "primary.main",
                },
              },
            }}
          />
        </Box>

        {/* Operation Toggle */}
        <Box sx={{ mt: 3, display: "flex", gap: 2 }}>
          <Button
            variant={operation === "encrypt" ? "contained" : "outlined"}
            onClick={() => setOperation("encrypt")}
            startIcon={<LockIcon />}
            fullWidth
            sx={{
              height: 48,
              fontSize: "1rem",
              fontWeight: "bold",
              borderRadius: 2,
            }}
          >
            Encrypt
          </Button>
          <Button
            variant={operation === "decrypt" ? "contained" : "outlined"}
            onClick={() => setOperation("decrypt")}
            startIcon={<LockOpenIcon />}
            fullWidth
            sx={{
              height: 48,
              fontSize: "1rem",
              fontWeight: "bold",
              borderRadius: 2,
            }}
          >
            Decrypt
          </Button>
        </Box>

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

        {/* Process Button */}
        <Button
          variant="contained"
          onClick={handleProcess}
          disabled={!selectedImage || !key || loading}
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
          {loading ? (
            <CircularProgress size={24} />
          ) : operation === "encrypt" ? (
            "Encrypt Image"
          ) : (
            "Decrypt Image"
          )}
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
          {operation === "encrypt" ? "Encryption" : "Decryption"} Results
        </Typography>

        {result ? (
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
                      <SecurityIcon
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
                      {operation === "encrypt"
                        ? "IMAGE ENCRYPTED"
                        : "IMAGE DECRYPTED"}
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
                        {result.processing_time.toFixed(2)}ms
                      </Typography>
                      <Typography variant="body1" sx={{ mb: 1 }}>
                        <strong>Security Level:</strong>
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={result.security_level * 100}
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
                        {(result.security_level * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ mt: 2 }}>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      {operation === "encrypt" ? "Encrypted" : "Decrypted"}{" "}
                      Image
                    </Typography>
                    <Box
                      component="img"
                      src={result.processed_image}
                      alt={operation === "encrypt" ? "Encrypted" : "Decrypted"}
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
              Select an image, enter a key, and click{" "}
              {operation === "encrypt" ? '"Encrypt Image"' : '"Decrypt Image"'}{" "}
              to see results
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default EncryptionDecryption;
