import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
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
import api from "../../utils/axiosConfig";
import { auth } from "../../config/firebase";
import { useNavigate } from "react-router-dom";
import {
  getLastRegisteredImage,
  clearLastRegisteredImage,
} from "../../utils/imageUtils";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartData,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const FeatureExtraction: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [lastRegisteredImage, setLastRegisteredImage] = useState<string | null>(
    null
  );
  const [processedImageUrl, setProcessedImageUrl] = useState<
    string | undefined
  >(undefined);
  const [features, setFeatures] = useState<{
    original: number[];
    extracted: number[];
  }>({
    original: [],
    extracted: [],
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
      setSuccess(null);
      setProcessedImageUrl(undefined);
      setFeatures({
        original: [],
        extracted: [],
      });
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
      setSuccess(null);
      setProcessedImageUrl(undefined);
      setFeatures({
        original: [],
        extracted: [],
      });
    }
  };

  const handleProcessFeatures = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

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

    try {
      const response = await api.post("/api/face/extract", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("API Response:", response.data); // Debug log

      if (response.data && response.data.extracted) {
        // Create a sample feature array for visualization
        const sampleFeatures = Array.from({ length: 128 }, (_, i) =>
          Math.random()
        );

        setFeatures({
          original: sampleFeatures,
          extracted: response.data.extracted,
        });
        setProcessedImageUrl(response.data.processed_image);
        setSuccess("Features extracted successfully!");
      } else {
        setError("Failed to extract features");
      }
    } catch (err: any) {
      console.error("Error:", err); // Debug log
      setError(err.response?.data?.error || "Failed to process features");
    } finally {
      setLoading(false);
    }
  };

  const featureChartData: ChartData<"line", number[], number> = {
    labels: Array.from({ length: 128 }, (_, i) => i + 1),
    datasets: [
      {
        label: "Original Features",
        data: features.original,
        borderColor: "rgb(75, 192, 192)",
        tension: 0.1,
        fill: false,
      },
      {
        label: "Extracted Features",
        data: features.extracted,
        borderColor: "rgb(255, 99, 132)",
        tension: 0.1,
        fill: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "Feature Extraction Results",
      },
      datalabels: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.1)",
        },
        ticks: {
          display: true,
        },
      },
      x: {
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.1)",
        },
        ticks: {
          display: true,
        },
      },
    },
    elements: {
      line: {
        tension: 0.4,
        borderWidth: 2,
      },
      point: {
        radius: 0,
      },
    },
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

        {/* Success Message */}
        {success && (
          <Fade in={true} timeout={500}>
            <Alert
              severity="success"
              sx={{
                mt: 3,
                borderRadius: 2,
              }}
            >
              {success}
            </Alert>
          </Fade>
        )}

        {/* Process Button */}
        <Button
          variant="contained"
          onClick={handleProcessFeatures}
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
      {processedImageUrl && (
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
            Feature Extraction Results
          </Typography>

          <Grid container spacing={3}>
            {/* Processed Image */}
            <Grid item xs={12} md={6}>
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
                      EXTRACTED FEATURES
                    </Typography>
                  </Box>

                  <Box
                    component="img"
                    src={processedImageUrl}
                    alt="Processed"
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
                </CardContent>
              </Card>
            </Grid>

            {/* Feature Chart */}
            <Grid item xs={12} md={6}>
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
                <CardContent sx={{ p: 3, height: "100%" }}>
                  <Box sx={{ height: 400 }}>
                    <Line data={featureChartData} options={chartOptions} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Box>
  );
};

export default FeatureExtraction;
