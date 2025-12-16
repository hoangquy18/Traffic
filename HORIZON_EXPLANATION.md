# Giải thích về Prediction Horizons

## Sự khác biệt giữa `horizon = 3` và `horizon = [1, 2, 3]`

### 1. `prediction_horizons: [3]` - Chỉ dự đoán 1 horizon

**Cấu hình:**
```yaml
prediction_horizons: [3]
```

**Cách hoạt động:**
- Model chỉ học để dự đoán **3 bước thời gian** trong tương lai
- Với mỗi sequence input, model chỉ tạo 1 prediction cho horizon 3
- Output shape: `[batch_size, 1, num_classes]`

**Ví dụ:**
```
Input sequence:  [t-7, t-6, t-5, t-4, t-3, t-2, t-1, t0]
Target:          [LOS tại t0+3]  ← chỉ có 1 giá trị
Prediction:      [LOS tại t0+3]  ← chỉ có 1 giá trị
```

**Metrics:**
- Chỉ có metrics cho horizon 3: `f1_macro_h3`, `accuracy_h3`, etc.

---

### 2. `prediction_horizons: [1, 2, 3]` - Dự đoán nhiều horizons cùng lúc

**Cấu hình:**
```yaml
prediction_horizons: [1, 2, 3]
```

**Cách hoạt động:**
- Model học để dự đoán **3 horizons khác nhau** cùng một lúc (multi-task learning)
- Với mỗi sequence input, model tạo 3 predictions: cho horizon 1, 2, và 3
- Output shape: `[batch_size, 3, num_classes]`

**Ví dụ:**
```
Input sequence:  [t-7, t-6, t-5, t-4, t-3, t-2, t-1, t0]
Targets:         [LOS tại t0+1, LOS tại t0+2, LOS tại t0+3]  ← 3 giá trị
Predictions:     [LOS tại t0+1, LOS tại t0+2, LOS tại t0+3]  ← 3 giá trị
```

**Metrics:**
- Có metrics cho tất cả 3 horizons:
  - `f1_macro_h1`, `accuracy_h1` (cho horizon 1)
  - `f1_macro_h2`, `accuracy_h2` (cho horizon 2)
  - `f1_macro_h3`, `accuracy_h3` (cho horizon 3)
- Metric chính (`f1_macro`) sử dụng horizon đầu tiên (h1)

---

## Lợi ích của `[1, 2, 3]` so với `[3]`

1. **Multi-task Learning**: Model học tốt hơn vì phải học cả short-term và long-term patterns
2. **Nhiều thông tin hơn**: Có thể đánh giá performance ở nhiều horizons khác nhau
3. **Transfer Learning**: Kiến thức từ horizon gần (h1, h2) giúp cải thiện horizon xa (h3)
4. **Linh hoạt**: Có thể sử dụng prediction cho bất kỳ horizon nào cần thiết

---

## Code Implementation

### Trong `sequential.py` (line 271-272):
```python
targets_vector = [
    los_array[end_idx - 1 + horizon] for horizon in prediction_horizons
]
```
→ Tạo targets cho tất cả horizons được chỉ định

### Trong `rnn.py` (line 76):
```python
nn.Linear(hidden_dim, num_classes * num_horizons)
```
→ Output layer tạo `num_classes * num_horizons` logits

### Trong `base.py` (line 258-298):
```python
for h_idx, horizon in enumerate(self.prediction_horizons):
    # Compute metrics for each horizon
```
→ Tính metrics riêng cho từng horizon

---

## Kết luận

- **`[3]`**: Đơn giản, chỉ cần prediction cho 3 bước tương lai
- **`[1, 2, 3]`**: Phức tạp hơn nhưng mạnh mẽ hơn, học được nhiều patterns hơn và cho nhiều thông tin hơn

Khuyến nghị: Sử dụng `[1, 2, 3]` để có model tốt hơn và đánh giá toàn diện hơn.

