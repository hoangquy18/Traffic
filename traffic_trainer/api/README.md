# Traffic Prediction API

FastAPI-based REST API for traffic LOS (Level of Service) prediction and street correlation analysis.

## Project Structure

```
traffic_trainer/api/
├── main.py              # FastAPI app entry point
├── config.py            # Configuration and paths
│
├── models/              # Pydantic schemas
│   ├── requests.py      # Request models
│   └── responses.py     # Response models
│
├── routes/              # API endpoints
│   ├── health.py        # Health check & model info
│   ├── admin.py         # Model management
│   ├── streets.py       # Street information
│   ├── prediction.py    # Traffic prediction
│   └── correlation.py   # Street correlation
│
├── core/                # Core logic
│   ├── model_loader.py  # Load PyTorch models
│   ├── inference.py     # Run predictions
│   ├── correlation.py   # Extract correlations
│   └── data_service.py  # Load/query traffic data
│
└── services/            # Service layer
    └── model_service.py # Main service facade
```

## Quick Start

### 1. Start the server

```bash
cd /path/to/T4
uvicorn traffic_trainer.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Load a model

```bash
# List available models
curl http://localhost:8000/models

# Load a model
curl -X POST http://localhost:8000/load-model \
  -H "Content-Type: application/json" \
  -d '{"model": "gman"}'
```

### 3. Make predictions

```bash
# Predict for a single segment
curl http://localhost:8000/predict/72693918077428

# Predict for multiple segments
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "segment_ids": [72693918077428, 206641347733512],
    "target_time": "2025-01-15T08:30:00"
  }'
```

### 4. Get correlations

```bash
# Get correlated streets
curl http://localhost:8000/correlations/72693918077428?top_k=10
```

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model-info` | Get loaded model info |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models` | List available models |
| POST | `/load-model` | Load a model |

### Streets

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/streets` | List streets (with search) |
| GET | `/streets/{segment_id}` | Get street info |

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict for multiple segments |
| GET | `/predict/{segment_id}` | Quick predict for one segment |

### Correlation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/correlations` | Get correlations with options |
| GET | `/correlations/{segment_id}` | Quick correlation lookup |

## Model Weights Structure

Place model weights in `weights/` folder:

```
weights/
├── gman/
│   ├── best_model.ckpt   # Required: model checkpoint
│   ├── config.yaml       # Optional: model config
│   └── metadata.json     # Optional: segment mappings
│
├── transformer/
│   └── ...
│
└── stgcn/
    └── ...
```

## Data File

Place traffic/weather data at project root:

```
traffic_weather_2025_converted.csv
```

Required columns:
- `segment_id`: Segment identifier
- `datetime_traffic`: Timestamp
- `street_name`: Street name
- Feature columns (numeric)

## Environment

```bash
# Create environment
conda create -n traffic python=3.11
conda activate traffic

# Install dependencies
pip install -e .
pip install fastapi uvicorn
```

## Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
