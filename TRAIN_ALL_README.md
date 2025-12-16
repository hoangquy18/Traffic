# Train All Models Script

Script để train tất cả models (ML và DL) với và không có weather features.

## Cách sử dụng

### Chạy tất cả (ML + DL, với và không có weather)
```bash
python train_all_models.py
```

### Chỉ chạy ML models
```bash
python train_all_models.py --ml-only
```

### Chỉ chạy DL models
```bash
python train_all_models.py --dl-only
```

### Chỉ chạy với weather features
```bash
python train_all_models.py --with-weather-only
```

### Chỉ chạy không có weather features
```bash
python train_all_models.py --no-weather-only
```

### Tùy chỉnh sequence_length và prediction_horizon
```bash
python train_all_models.py --min-seq-len 1 --max-seq-len 6 --min-horizon 1 --max-horizon 5
```

## Kết quả

Kết quả sẽ được lưu vào:

### ML Models:
- `experiments/ml_scan_with_weather.csv` - Với weather features
- `experiments/ml_scan_no_weather.csv` - Không có weather features

### DL Models:
- `experiments/dl_scan_with_weather.csv` - Với weather features
- `experiments/dl_scan_no_weather.csv` - Không có weather features

## Models được train

### ML Models:
- Decision Tree
- XGBoost

### DL Models:
- RNN
- GNN
- Transformer
- TCN
- Informer
- TimesNet
- GMAN++

## Lưu ý

- Script sẽ chạy tuần tự từng phần (ML with weather → ML no weather → DL with weather → DL no weather)
- Mỗi phần có thể mất nhiều thời gian tùy vào số lượng hyperparameters và combinations
- Dataset sẽ được cache để tăng tốc độ (giống như trong dl_scan.py và ml_grid_search.py)
- Có thể dừng script bằng Ctrl+C, các phần đã hoàn thành sẽ được lưu

