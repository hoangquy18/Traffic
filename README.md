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
  - **TimesNet** - Temporal 2D-Variation Modeling using 2D FFT transformations
  - **TimesNet++** - Enhanced TimesNet with multi-scale 2D FFT and cross-scale fusion
  - **Informer** - Efficient Transformer for long sequence time-series forecasting
  - **TCN** - Temporal Convolutional Network with multi-scale convolutions
  - **XGBoost** - Gradient boosting for classification
  - **Decision Tree** - Tree-based classifier
  - **ARIMA** - Autoregressive Integrated Moving Average
  - **SARIMA** - Seasonal ARIMA with seasonal components
- Early stopping, learning-rate scheduling, checkpointing, and optional Weights & Biases logging.

### Project Layout

```
traffic_trainer/
├── __init__.py              # Main package exports
├── configs/                  # Configuration files
│   ├── config.yaml          # RNN model config
│   ├── graph_config.yaml    # GNN model config
│   ├── transformer_config.yaml
│   ├── sota_config.yaml     # GMAN config
│   ├── timesnet_config.yaml
│   ├── timesnet_plus_plus_config.yaml
│   ├── informer_config_optimized.yaml
│   ├── tcn_config.yaml
│   ├── xgboost_config.yaml
│   ├── decision_tree_config.yaml
│   ├── arima_config.yaml
│   └── sarima_config.yaml
├── data/                     # Data loading modules
│   ├── constants.py         # LOS_LEVELS and shared constants
│   ├── sequential.py        # Sequential data loader (for RNN)
│   └── graph.py             # Graph/Transformer data loader
├── models/                   # Model architectures
│   ├── rnn.py               # SequenceClassifier (LSTM/GRU)
│   ├── gnn.py               # SpatioTemporalGNN (GCN/GAT)
│   ├── transformer.py       # SpatioTemporalTransformer
│   ├── gman.py              # GMAN++ model
│   ├── timesnet.py          # TimesNet model
│   ├── timesnet_plus_plus.py # TimesNet++ model
│   ├── informer.py          # Informer model
│   └── tcn.py               # Temporal Convolutional Network
├── trainers/                 # Training scripts
│   ├── base.py              # Base trainer for deep learning models
│   ├── ml_base.py           # Base trainer for traditional ML models
│   ├── rnn_trainer.py       # RNN trainer
│   ├── gnn_trainer.py       # GNN trainer
│   ├── transformer_trainer.py
│   ├── gman_trainer.py      # GMAN trainer
│   ├── timesnet_trainer.py  # TimesNet trainer
│   ├── timesnet_plus_plus_trainer.py # TimesNet++ trainer
│   ├── informer_trainer.py  # Informer trainer
│   ├── tcn_trainer.py       # TCN trainer
│   ├── xgboost_trainer.py   # XGBoost trainer
│   ├── decision_tree_trainer.py # Decision Tree trainer
│   ├── arima_trainer.py     # ARIMA trainer
│   └── sarima_trainer.py    # SARIMA trainer
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
- For ML models: `xgboost`, `statsmodels`

Install dependencies manually, for example:

```bash
conda create -n traffic python=3.10
conda activate traffic
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # pick the wheel that matches your setup
pip install pandas numpy scikit-learn pyyaml tqdm wandb joblib xgboost statsmodels
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

#### 5. TimesNet

Best for: Capturing both intraperiod and interperiod variations using 2D FFT transformations.

```bash
python -m traffic_trainer.trainers.timesnet_trainer --config traffic_trainer/configs/timesnet_config.yaml
```

**Key features:**
- 2D FFT transformation converts 1D time series to 2D representation
- Inception-like blocks for multi-scale feature extraction
- Captures temporal variations at different periods

**Config options:**
- `timesnet.top_k`: Number of top frequencies to keep
- `timesnet.e_layers`: Number of encoder layers
- `timesnet.num_kernels`: Number of kernels in Inception block
- `timesnet.d_ff`: Feed-forward dimension

#### 6. TimesNet++

Best for: Enhanced temporal modeling with multi-scale analysis and cross-scale fusion.

```bash
python -m traffic_trainer.trainers.timesnet_plus_plus_trainer --config traffic_trainer/configs/timesnet_plus_plus_config.yaml
```

**Enhanced features:**
- Multi-scale 2D FFT with multiple period detection
- Enhanced Inception blocks with channel attention
- Cross-scale feature fusion for better integration
- Adaptive period selection

**Config options:**
- `timesnet_plus_plus.top_k`: Number of top frequencies
- `timesnet_plus_plus.num_periods`: Number of different periods for multi-scale analysis
- `timesnet_plus_plus.e_layers`: Number of encoder layers

#### 7. Informer

Best for: Efficient long sequence time-series forecasting with ProbSparse attention.

```bash
python -m traffic_trainer.trainers.informer_trainer --config traffic_trainer/configs/informer_config_optimized.yaml
```

**Key features:**
- ProbSparse Self-Attention (O(L log L) complexity)
- Self-Attention Distilling for dimension reduction
- Generative style decoder for long sequence prediction

#### 8. TCN (Temporal Convolutional Network)

Best for: Multi-scale temporal pattern recognition with causal convolutions.

```bash
python -m traffic_trainer.trainers.tcn_trainer --config traffic_trainer/configs/tcn_config.yaml
```

**Key features:**
- Causal convolutions for temporal modeling
- Multi-scale temporal blocks
- Dilated convolutions for capturing long-range dependencies

#### 9. XGBoost

Best for: Fast, interpretable gradient boosting with good baseline performance.

```bash
python -m traffic_trainer.trainers.xgboost_trainer --config traffic_trainer/configs/xgboost_config.yaml
```

**Key features:**
- Gradient boosting for classification
- Uses current time step features to predict future horizons
- One model per prediction horizon
- Fast training and inference

**Config options:**
- `xgboost.n_estimators`: Number of boosting rounds
- `xgboost.max_depth`: Maximum tree depth
- `xgboost.learning_rate`: Learning rate
- `xgboost.subsample`: Row subsampling ratio

#### 10. Decision Tree

Best for: Simple, interpretable baseline model.

```bash
python -m traffic_trainer.trainers.decision_tree_trainer --config traffic_trainer/configs/decision_tree_config.yaml
```

**Key features:**
- Simple tree-based classifier
- Fast training and inference
- Interpretable decision rules
- One model per prediction horizon

**Config options:**
- `decision_tree.max_depth`: Maximum tree depth (null for unlimited)
- `decision_tree.min_samples_split`: Minimum samples to split
- `decision_tree.criterion`: Split criterion ("gini" or "entropy")

#### 11. ARIMA

Best for: Univariate time series forecasting with autoregressive components.

```bash
python -m traffic_trainer.trainers.arima_trainer --config traffic_trainer/configs/arima_config.yaml
```

**Key features:**
- Autoregressive Integrated Moving Average model
- Works with aggregated time series across segments
- Suitable for univariate forecasting

**Config options:**
- `arima.order`: ARIMA order (p, d, q)
- `arima.max_iter`: Maximum iterations for model fitting

#### 12. SARIMA

Best for: Time series with seasonal patterns (e.g., hourly traffic data).

```bash
python -m traffic_trainer.trainers.sarima_trainer --config traffic_trainer/configs/sarima_config.yaml
```

**Key features:**
- Seasonal ARIMA with seasonal components
- Captures daily/hourly patterns
- Default seasonal period of 24 for hourly data

**Config options:**
- `sarima.order`: ARIMA order (p, d, q)
- `sarima.seasonal_order`: Seasonal order (P, D, Q, s) where s is the seasonal period
- `sarima.max_iter`: Maximum iterations for model fitting

---

### Training Output

Inside `paths.output_dir` you will find:

| File | Description |
|------|-------------|
| `best_model.pt` | Best validation weights for deep learning models (optimizer state optional) |
| `checkpoint_epoch*.pt` | Periodic checkpoints for deep learning models |
| `model_h*.joblib` | Trained models for ML models (one per horizon) |
| `test_results.json` | Test metrics and classification report |
| `history.json` | Training history (loss, F1, accuracy per epoch) |
| `scaler.joblib` | Fitted StandardScaler for inference |
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
| **TimesNet** | Temporal patterns | Captures intra/interperiod variations | Requires tuning top_k |
| **TimesNet++** | Enhanced temporal | Multi-scale analysis | More complex |
| **Informer** | Long sequences | Efficient attention | Less interpretable |
| **TCN** | Multi-scale patterns | Causal convolutions | Fixed receptive field |
| **XGBoost** | Baseline/Interpretable | Fast, good performance | No temporal modeling |
| **Decision Tree** | Simple baseline | Very fast, interpretable | Limited capacity |
| **ARIMA** | Univariate TS | Classical approach | Univariate only |
| **SARIMA** | Seasonal patterns | Captures seasonality | Requires long sequences |

---
