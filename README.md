## Traffic Trainer

Sequence modelling pipeline for predicting traffic Level of Service (LOS) categories from historical traffic and weather observations. The project packages data preparation, model definition, training, and evaluation utilities under `traffic_trainer/`.

### Features

- Chronological dataset split (train/val/test) with rolling sequence generation.
- Normalisation pipeline that is robust against zero-variance features and missing data after resampling.
- Multiple model architectures:
  - **RNN Classifier** (GRU/LSTM, uni- or bi-directional)
  - **Spatio-Temporal GNN** (GCN/GAT) for traffic propagation modeling
  - **Spatio-Temporal Transformer** for learning segment relationships via self-attention
  - **GMAN++** (Graph Multi-Attention Network) with dilated temporal convolutions
- Early stopping, learning-rate scheduling, checkpointing, and optional Weights & Biases logging.

### Project Layout

```
traffic_trainer/
├── __init__.py              # Main package exports
├── configs/                  # Configuration files
│   ├── config.yaml          # RNN model config
│   ├── graph_config.yaml    # GNN model config
│   ├── transformer_config.yaml
│   └── sota_config.yaml     # GMAN config
├── data/                     # Data loading modules
│   ├── constants.py         # LOS_LEVELS and shared constants
│   ├── sequential.py        # Sequential data loader (for RNN)
│   └── graph.py             # Graph/Transformer data loader
├── models/                   # Model architectures
│   ├── rnn.py               # SequenceClassifier (LSTM/GRU)
│   ├── gnn.py               # SpatioTemporalGNN (GCN/GAT)
│   ├── transformer.py       # SpatioTemporalTransformer
│   └── gman.py              # GMAN++ model
├── trainers/                 # Training scripts
│   ├── rnn_trainer.py       # RNN trainer
│   ├── gnn_trainer.py       # GNN trainer
│   ├── transformer_trainer.py
│   └── gman_trainer.py      # GMAN trainer
└── utils/                    # Utilities
    ├── feature_importance.py
    └── preprocessing.py     # Data preprocessing utilities
```

Additional directories:
- `experiments/` — default location for checkpoints and logs (created automatically).
- `traffic_weather_data/` — raw daily CSV files from TomTom API.

### Requirements

- Python 3.9+
- PyTorch (CPU or CUDA build, depending on your hardware)
- pandas
- numpy
- scikit-learn
- PyYAML
- joblib
- Optional: `wandb` for experiment tracking

Install dependencies manually, for example:

```bash
conda create -n traffic python=3.10
conda activate traffic
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # pick the wheel that matches your setup
pip install pandas numpy scikit-learn pyyaml tqdm wandb joblib
```

### Dataset Expectations & Preparation

The loader expects a CSV with:

- Timestamp column `datetime_traffic` (timezone-aware ISO format recommended)
- An LOS label column `LOS` containing one of `A`–`F`
- Feature columns referenced in the config (`numerical_features`, `categorical_features`)
- Segment identifier column `segment_id`

During preprocessing:

1. Categorical features are one-hot encoded.
2. Numerical & encoded features are forward/backward filled, then standardised (zero-variance columns are handled safely).
3. Each segment is resampled (default `1H`) with time interpolation plus forward/backward fill.
4. Rolling windows of length `sequence_length` are generated and labelled with the LOS at the final timestep.
5. Windows are assigned to splits based on chronological cutoffs computed from the global timestamp distribution (`train_ratio`, `val_ratio`, remainder for test).

---

### Preprocessing New Data

Raw data dumps from TomTom API (stored in `traffic_weather_data/`) need to be converted to the unified training format. Use the preprocessing script:

```bash
python -m traffic_trainer.utils.preprocessing \
  --input-dir traffic_weather_data \
  --output-path traffic_weather_2025_converted.csv
```

**What the script does:**
- Reads all CSV files from the input directory
- Computes LOS (Level of Service) based on speed ratios and velocity thresholds
- Merges traffic data with weather observations
- Outputs a unified CSV ready for training

**LOS Classification Thresholds:**
| Speed Limit | LOS A | LOS B | LOS C | LOS D | LOS E | LOS F |
|-------------|-------|-------|-------|-------|-------|-------|
| 40-50 km/h  | ≥35   | 30-35 | 25-30 | 20-25 | 15-20 | <15   |
| 60 km/h     | ≥40   | 35-40 | 30-35 | 25-30 | 20-25 | <20   |
| 70 km/h     | ≥45   | 40-45 | 35-40 | 30-35 | 25-30 | <25   |

Once generated, update `paths.csv_path` in your config to point to the converted file.

---

### Configuration

Edit the appropriate config file in `traffic_trainer/configs/` or supply your own via `--config`. Key sections:

- `paths.csv_path` — absolute path to the input CSV.
- `paths.output_dir` — directory where logs, checkpoints, and metrics are stored.
- `data` — feature lists, sequence length, split ratios, resample rule.
- `model` — RNN type, hidden size, layers, dropout, bidirectionality.
- `model.time_embedding_dim` — optional learned embedding size for hour-of-day (set to `null` to disable).
- `model.segment_embedding_dim` — optional learned embedding size for segment IDs (set to `null` to disable).
- `optimization` — batch size, learning rate, weight decay, gradient clipping.
- `training` — epochs, dataloader workers, target device (`cpu` or `cuda`).
- `logging` — wandb project/entity/run name and mode (`online`, `offline`, `disabled`).
- `early_stopping` / `checkpoint` — patience and checkpoint cadence.

---

### Running Training

#### 1. Sequential RNN Model

Best for: Independent segment prediction, fast training.

```bash
python -m traffic_trainer.trainers.rnn_trainer --config traffic_trainer/configs/config.yaml
```

- Supports LSTM or GRU with optional bidirectional processing
- Optional time and segment embeddings
- Multi-horizon prediction support

#### 2. Graph Neural Network (STGNN)

Best for: Modeling traffic propagation between connected road segments.

```bash
python -m traffic_trainer.trainers.gnn_trainer --config traffic_trainer/configs/graph_config.yaml
```

**How the graph is built:**
- Each road segment becomes a **node** in the graph
- Two segments are **connected** if one segment's end node (`e_node_id`) equals another segment's start node (`s_node_id`)
- This captures how traffic flows and propagates through the road network

**Architecture:**
1. **Temporal Encoder (GRU/LSTM)**: Processes time series for each road segment
2. **Spatial Encoder (GCN/GAT)**: Aggregates information from neighboring segments
3. **Classification Head**: Predicts LOS for each segment

**Config options:**
- `graph.gnn_type`: `"gcn"` or `"gat"` (Graph Attention Network)
- `graph.num_gnn_layers`: Number of graph convolution layers
- `graph.gat_heads`: Number of attention heads for GAT

#### 3. Spatio-Temporal Transformer

Best for: Learning segment relationships without explicit graph topology.

```bash
python -m traffic_trainer.trainers.transformer_trainer --config traffic_trainer/configs/transformer_config.yaml
```

- Uses self-attention to learn which segments influence each other
- No need for `s_node_id`/`e_node_id` columns
- Learnable segment embeddings

#### 4. GMAN++ (Graph Multi-Attention Network)

Best for: State-of-the-art multi-horizon prediction with advanced features.

```bash
python -m traffic_trainer.trainers.gman_trainer --config traffic_trainer/configs/sota_config.yaml
```

**Advanced features:**
- Dilated temporal convolutions for multi-scale patterns
- Autoregressive decoder with horizon embeddings
- Ordinal focal loss for LOS prediction (respects A < B < C < D < E < F ordering)
- Optional class weighting for imbalanced data

**Config options:**
- `gman.num_heads`: Number of attention heads
- `gman.use_spatial_embedding`: Learnable segment identity embeddings
- `gman.use_temporal_conv`: Enable dilated temporal convolutions
- `use_ordinal_loss`: Penalize predictions based on ordinal distance

---

### Training Output

Inside `paths.output_dir` you will find:

| File | Description |
|------|-------------|
| `best_model.pt` | Best validation weights (optimizer state optional) |
| `checkpoint_epoch*.pt` | Periodic checkpoints |
| `metrics.json` / `test_results.json` | Test metrics and classification report |
| `history.json` | Training history (loss, F1, accuracy per epoch) |
| `scaler.joblib` / `scaler.pkl` | Fitted StandardScaler for inference |
| `feature_names.json` | Ordered feature list |
| `metadata.json` | Segment mappings and graph info |
| `confusion_matrix_*.png` | Confusion matrices (RNN trainer) |

---

### Feature Importance Analysis

Compute gradient-based feature importance for a trained model:

```bash
python -m traffic_trainer.utils.feature_importance \
  --config traffic_trainer/configs/config.yaml \
  --checkpoint experiments/run01/best_model.pt \
  --split val \
  --device cuda \
  --output-csv importance.csv
```

---

### Model Selection Guide

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| **RNN** | Independent segments | Fast, simple | No spatial modeling |
| **GNN** | Connected road network | Models traffic propagation | Requires topology data |
| **Transformer** | Any multi-segment data | Learns relationships automatically | More parameters |
| **GMAN++** | Production/SOTA | Best accuracy, multi-horizon | Slowest training |

---
