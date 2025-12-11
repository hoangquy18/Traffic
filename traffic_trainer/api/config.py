"""API configuration and default paths."""

from pathlib import Path

# Default paths
WEIGHTS_DIR = Path("weights")
DATA_FILE = Path("traffic_weather_2025_converted.csv")

# API settings
API_TITLE = "Traffic Prediction API"
API_DESCRIPTION = "API for traffic LOS prediction and street correlation analysis"
API_VERSION = "1.0.0"
