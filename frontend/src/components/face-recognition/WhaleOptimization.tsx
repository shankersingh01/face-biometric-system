import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
} from "@mui/material";
import { CloudUpload as CloudUploadIcon } from "@mui/icons-material";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import { getAuthToken } from "../../utils/auth";
import ChartDataLabels from "chartjs-plugin-datalabels";
import {
  getLastRegisteredImage,
  clearLastRegisteredImage,
} from "../../utils/imageUtils";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartDataLabels
);

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`optimization-tabpanel-${index}`}
      aria-labelledby={`optimization-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const WhaleOptimization: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [lastRegisteredImage, setLastRegisteredImage] = useState<string | null>(
    null
  );
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [optimizationData, setOptimizationData] = useState<{
    extracted: number[];
    optimized: number[];
    performance: {
      extractionTime: number;
      optimizationTime: number;
      accuracy: number;
      storageEfficiency: {
        originalSize: number;
        optimizedSize: number;
        reductionPercentage: number;
      };
      featureQuality: {
        originalVariance: number;
        optimizedVariance: number;
        distinctivenessImprovement: number;
      };
      robustness: {
        originalRobustness: number;
        optimizedRobustness: number;
        improvementPercentage: number;
      };
      matchingSpeed: {
        originalMatchingTime: number;
        optimizedMatchingTime: number;
        improvementPercentage: number;
      };
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
      setOptimizationData(null);
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
      setOptimizationData(null);
    }
  };

  const handleOptimize = async () => {
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
        "http://127.0.0.1:5000/api/face/optimize",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setOptimizationData(response.data);
    } catch (error) {
      console.error("Optimization error:", error);
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

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const featureChartData = optimizationData
    ? {
        labels: Array.from(
          { length: optimizationData.extracted.length },
          (_, i) => i + 1
        ),
        datasets: [
          {
            label: "Extracted Features",
            data: optimizationData.extracted,
            borderColor: "rgb(75, 192, 192)",
            tension: 0.1,
          },
          {
            label: "Optimized Features",
            data: optimizationData.optimized,
            borderColor: "rgb(255, 99, 132)",
            tension: 0.1,
          },
        ],
      }
    : null;

  const performanceChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "Performance Metrics",
      },
      datalabels: {
        anchor: "end" as const,
        align: "top" as const,
        formatter: (value: number) => value.toFixed(2),
        font: {
          weight: "bold" as const,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const featureChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "Feature Optimization Comparison",
      },
      datalabels: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const renderPerformanceMetrics = () => {
    if (!optimizationData) return null;

    const {
      extractionTime,
      optimizationTime,
      storageEfficiency,
      featureQuality,
      robustness,
      matchingSpeed,
      accuracy,
    } = optimizationData.performance;

    // Processing Times Chart
    const processingTimesData = {
      labels: ["Feature Extraction", "Whale Optimization"],
      datasets: [
        {
          label: "Processing Time (ms)",
          data: [extractionTime, optimizationTime],
          backgroundColor: [
            "rgba(54, 162, 235, 0.5)",
            "rgba(255, 99, 132, 0.5)",
          ],
          borderColor: ["rgba(54, 162, 235, 1)", "rgba(255, 99, 132, 1)"],
          borderWidth: 1,
        },
      ],
    };

    // Storage Efficiency Chart
    const storageData = {
      labels: ["Original", "Optimized"],
      datasets: [
        {
          label: "Size (KB)",
          data: [
            storageEfficiency.originalSize / 1024,
            storageEfficiency.optimizedSize / 1024,
          ],
          backgroundColor: [
            "rgba(75, 192, 192, 0.5)",
            "rgba(153, 102, 255, 0.5)",
          ],
          borderColor: ["rgba(75, 192, 192, 1)", "rgba(153, 102, 255, 1)"],
          borderWidth: 1,
        },
      ],
    };

    // Feature Quality Chart
    const featureQualityData = {
      labels: ["Original", "Optimized"],
      datasets: [
        {
          label: "Variance",
          data: [
            featureQuality.originalVariance,
            featureQuality.optimizedVariance,
          ],
          backgroundColor: [
            "rgba(255, 159, 64, 0.5)",
            "rgba(255, 205, 86, 0.5)",
          ],
          borderColor: ["rgba(255, 159, 64, 1)", "rgba(255, 205, 86, 1)"],
          borderWidth: 1,
        },
      ],
    };

    // Robustness Chart
    const robustnessData = {
      labels: ["Original", "Optimized"],
      datasets: [
        {
          label: "Robustness Score",
          data: [robustness.originalRobustness, robustness.optimizedRobustness],
          backgroundColor: [
            "rgba(201, 203, 207, 0.5)",
            "rgba(54, 162, 235, 0.5)",
          ],
          borderColor: ["rgba(201, 203, 207, 1)", "rgba(54, 162, 235, 1)"],
          borderWidth: 1,
        },
      ],
    };

    // Matching Speed Chart
    const matchingSpeedData = {
      labels: ["Original", "Optimized"],
      datasets: [
        {
          label: "Matching Time (ms)",
          data: [
            matchingSpeed.originalMatchingTime,
            matchingSpeed.optimizedMatchingTime,
          ],
          backgroundColor: [
            "rgba(255, 99, 132, 0.5)",
            "rgba(75, 192, 192, 0.5)",
          ],
          borderColor: ["rgba(255, 99, 132, 1)", "rgba(75, 192, 192, 1)"],
          borderWidth: 1,
        },
      ],
    };

    // Overall Performance Gauge
    const gaugeData = {
      labels: ["Accuracy"],
      datasets: [
        {
          data: [accuracy],
          backgroundColor: ["rgba(75, 192, 192, 0.5)"],
          borderColor: ["rgba(75, 192, 192, 1)"],
          borderWidth: 1,
        },
      ],
    };

    return (
      <div className="space-y-6">
        {/* Overall Accuracy Gauge */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Overall Performance</h3>
          <div className="flex justify-center items-center h-64">
            <div className="text-center">
              <div className="text-4xl font-bold text-green-600 mb-2">
                {accuracy.toFixed(2)}%
              </div>
              <div className="text-gray-600">Accuracy Score</div>
            </div>
          </div>
        </div>

        {/* Processing Times */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Processing Times</h3>
          <div className="h-64">
            <Bar data={processingTimesData} options={performanceChartOptions} />
          </div>
        </div>

        {/* Storage Efficiency */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Storage Efficiency</h3>
          <div className="h-64">
            <Bar data={storageData} options={performanceChartOptions} />
          </div>
        </div>

        {/* Feature Quality */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Feature Quality</h3>
          <div className="h-64">
            <Bar data={featureQualityData} options={performanceChartOptions} />
          </div>
        </div>

        {/* Robustness */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Feature Robustness</h3>
          <div className="h-64">
            <Bar data={robustnessData} options={performanceChartOptions} />
          </div>
        </div>

        {/* Matching Speed */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Matching Speed</h3>
          <div className="h-64">
            <Bar data={matchingSpeedData} options={performanceChartOptions} />
          </div>
        </div>
      </div>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography
        variant="h4"
        gutterBottom
        sx={{ mb: 4, color: "primary.main" }}
      >
        Whale Optimization Algorithm
      </Typography>

      <Grid container spacing={4}>
        {/* Left Column - Image Selection and Controls */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3, height: "100%" }}>
            <Typography
              variant="h6"
              gutterBottom
              sx={{ color: "primary.main" }}
            >
              Image Selection
            </Typography>

            {/* Last Registered Image Section */}
            {lastRegisteredImage && (
              <Box
                sx={{
                  mb: 3,
                  p: 2,
                  border: "1px dashed",
                  borderColor: "primary.main",
                  borderRadius: 1,
                }}
              >
                <Typography
                  variant="subtitle1"
                  gutterBottom
                  sx={{ color: "text.secondary" }}
                >
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
                    borderRadius: 1,
                    boxShadow: 1,
                  }}
                />
                <Button
                  variant="contained"
                  onClick={handleUseLastRegisteredImage}
                  fullWidth
                  sx={{ mb: 2 }}
                >
                  Use This Image
                </Button>
              </Box>
            )}

            {/* Upload New Image Section */}
            <Box
              sx={{
                border: "2px dashed",
                borderColor: "primary.main",
                borderRadius: 1,
                p: 3,
                textAlign: "center",
                cursor: "pointer",
                mb: 3,
                "&:hover": {
                  backgroundColor: "action.hover",
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
                  sx={{ mb: 2 }}
                >
                  Upload New Image
                </Button>
              </label>
              <Typography variant="body2" color="text.secondary">
                or drag and drop an image here
              </Typography>
            </Box>

            {/* Selected Image Preview */}
            {selectedImage && (
              <Box sx={{ mb: 3 }}>
                <Typography
                  variant="subtitle1"
                  gutterBottom
                  sx={{ color: "text.secondary" }}
                >
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
                    borderRadius: 1,
                    boxShadow: 2,
                  }}
                />
              </Box>
            )}

            {/* Error Message */}
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {/* Optimize Button */}
            <Button
              variant="contained"
              onClick={handleOptimize}
              disabled={!selectedImage || loading}
              fullWidth
              size="large"
              sx={{
                height: 48,
                fontSize: "1.1rem",
                fontWeight: "bold",
              }}
            >
              {loading ? <CircularProgress size={24} /> : "Optimize Features"}
            </Button>
          </Paper>
        </Grid>

        {/* Right Column - Results */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3, height: "100%" }}>
            <Typography
              variant="h6"
              gutterBottom
              sx={{ color: "primary.main" }}
            >
              Optimization Results
            </Typography>

            {optimizationData ? (
              <Box sx={{ mt: 2 }}>
                <Tabs
                  value={tabValue}
                  onChange={handleTabChange}
                  sx={{ mb: 3 }}
                >
                  <Tab label="Feature Comparison" />
                  <Tab label="Performance Metrics" />
                </Tabs>

                <TabPanel value={tabValue} index={0}>
                  {featureChartData && (
                    <Box sx={{ height: 400 }}>
                      <Line
                        data={featureChartData}
                        options={featureChartOptions}
                      />
                    </Box>
                  )}
                </TabPanel>

                <TabPanel value={tabValue} index={1}>
                  {renderPerformanceMetrics()}
                </TabPanel>
              </Box>
            ) : (
              <Box
                sx={{
                  height: 400,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  bgcolor: "background.default",
                  borderRadius: 1,
                }}
              >
                <Typography color="text.secondary">
                  Select an image and click "Optimize Features" to see results
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default WhaleOptimization;
