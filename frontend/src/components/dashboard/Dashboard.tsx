import React, { useState, useEffect, useCallback } from "react";
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Button,
} from "@mui/material";
import {
  People as PeopleIcon,
  Security as SecurityIcon,
  Lock as LockIcon,
  VerifiedUser as VerifiedUserIcon,
} from "@mui/icons-material";
import { useAuth } from "../../contexts/AuthContext";
import { db } from "../../config/firebase";
import { collection, doc, onSnapshot, DocumentData } from "firebase/firestore";

interface DashboardStats {
  registeredFaces: number;
  encryptedFeatures: number;
  optimizationRate: number;
  matchAccuracy: number;
  lastUpdated: string;
  recentActivity: string[];
  systemStatus: {
    database: string;
    encryption: string;
    optimization: string;
    faceRecognition: string;
  };
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const { currentUser } = useAuth();

  const fetchDashboardData = useCallback(async () => {
    if (!currentUser) {
      setError("User not authenticated");
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const token = await currentUser.getIdToken();
      const response = await fetch(
        "http://127.0.0.1:5000/api/dashboard/stats",
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to load dashboard data");
      }

      const data = await response.json();
      setStats(data);
      setLastUpdate(new Date());
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load dashboard data"
      );
    } finally {
      setLoading(false);
    }
  }, [currentUser]);

  // Fetch data on component mount and when user changes
  useEffect(() => {
    if (currentUser) {
      fetchDashboardData();
    }
  }, [currentUser, fetchDashboardData]);

  // Listen for changes in Firestore
  useEffect(() => {
    if (!currentUser) return;

    const unsubscribe = onSnapshot(
      collection(db, "dashboard"),
      (snapshot) => {
        snapshot.docChanges().forEach((change) => {
          if (change.type === "modified") {
            const docData = change.doc.data() as DocumentData;
            if (docData.lastUpdated !== lastUpdate) {
              fetchDashboardData();
            }
          }
        });
      },
      (error) => {
        console.error("Error listening to dashboard updates:", error);
      }
    );

    return () => unsubscribe();
  }, [currentUser, lastUpdate, fetchDashboardData]);

  const statCards = [
    {
      title: "Registered Faces",
      value: stats?.registeredFaces?.toString() || "0",
      icon: <PeopleIcon sx={{ fontSize: 40 }} />,
      color: "#1976d2",
    },
    {
      title: "Encrypted Features",
      value: stats?.encryptedFeatures?.toString() || "0",
      icon: <SecurityIcon sx={{ fontSize: 40 }} />,
      color: "#2e7d32",
    },
    {
      title: "Optimization Rate",
      value: stats?.optimizationRate ? `${stats.optimizationRate}%` : "0%",
      icon: <LockIcon sx={{ fontSize: 40 }} />,
      color: "#ed6c02",
    },
    {
      title: "Match Accuracy",
      value: stats?.matchAccuracy ? `${stats.matchAccuracy}%` : "0%",
      icon: <VerifiedUserIcon sx={{ fontSize: 40 }} />,
      color: "#9c27b0",
    },
  ];

  if (loading && !stats) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="80vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="80vh"
      >
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard Overview
      </Typography>
      <Grid container spacing={3}>
        {statCards.map((stat) => (
          <Grid item xs={12} sm={6} md={3} key={stat.title}>
            <Card>
              <CardContent>
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                  }}
                >
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      {stat.title}
                    </Typography>
                    <Typography variant="h4">{stat.value}</Typography>
                  </Box>
                  <Box sx={{ color: stat.color }}>{stat.icon}</Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Activity
            </Typography>
            {stats?.recentActivity?.map((activity, index) => (
              <Typography key={index} variant="body2" color="textSecondary">
                • {activity}
              </Typography>
            ))}
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <Typography variant="body2" color="textSecondary">
              • Database: {stats?.systemStatus?.database || "Checking..."}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              • Encryption: {stats?.systemStatus?.encryption || "Checking..."}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              • Optimization:{" "}
              {stats?.systemStatus?.optimization || "Checking..."}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              • Face Recognition:{" "}
              {stats?.systemStatus?.faceRecognition || "Checking..."}
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
