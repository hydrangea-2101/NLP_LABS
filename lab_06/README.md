# NLP LAB 06 — ATTENTION MECHANISM
## 1. Tổng quan và Mục tiêu bài Lab
LAB 06 đánh dấu bước chuyển quan trọng trong xử lý chuỗi: từ việc nén toàn bộ ngữ cảnh vào một vector trạng thái ẩn duy nhất ($h_T$ trong RNN/LSTM) sang cơ chế biểu diễn nhận biết ngữ cảnh đa chiều (**Context-Aware Representation**) thông qua **Attention Mechanism**.

## 2. Cấu trúc thư mục Deliverables

Toàn bộ các tệp yêu cầu tại **Mục 32 — W6.pdf** được tổ chức đầy đủ và chuẩn mực như sau:

```text
lab_06/
├── caculations_lab06.pdf          # Báo cáo giải tay chi tiết các bài toán lý thuyết
├── prediction.md                  # 5 câu hỏi dự đoán trước thực nghiệm 
├── attention_numpy.py             # Cài đặt Attention cốt lõi bằng NumPy 
├── self_attention.ipynb           # Notebook Jupyter 13 phần hoàn chỉnh
├── results.csv                    # Bảng số liệu đối chiếu định lượng đo đạc từ thực nghiệm 
├── attention_heatmap.png          # Biểu đồ Heatmap trực quan hóa Attention Matrix 
├── confusion_matrix.png           # Ma trận nhầm lẫn của mô hình phân loại cảm xúc trên Test
├── error_analysis.md              # Báo cáo phân tích sâu 3 ca đúng và 3 ca sai thực tế kèm giả thuyết
├── reflection.md                  # Khung phản tư học tập, củng cố lý thuyết và nhật ký cá nhân
├── requirements.txt               # Danh sách thư viện Python cần thiết
└── README.md                      # Hướng dẫn chi tiết toàn diện bài lab
```

---

## 3. Môi trường và Cài đặt

### Yêu cầu môi trường
- Python $\ge$ 3.10 (Môi trường thực tế dự án: `Python 3.13.14`).
- Các gói thư viện: `numpy`, `pandas`, `torch` (CPU hoặc CUDA), `scikit-learn`, `matplotlib`, `nbformat`, `nbconvert`, `ipykernel`.

### Cài đặt thư viện
Nếu sử dụng môi trường virtualenv chung của dự án (`NLP_LABS\.venv`):
```bash
# Kích hoạt môi trường (Windows PowerShell)
..\.venv\Scripts\Activate.ps1

# Hoặc cài đặt các gói theo requirements.txt nếu tạo môi trường mới
pip install -r requirements.txt
```

---

## 4. Hướng dẫn chạy và Kiểm tra từng phần

### 4.1. Kiểm thử phần cài đặt Attention bằng NumPy (`attention_numpy.py`)
Tệp này cài đặt:
1. `stable_softmax(x, axis=-1)`: Đảm bảo ổn định số học qua phép trừ $\max(x)$ trên từng hàng, không bị tràn số với điểm số lớn (như $x = [1000, 1001, 1002]$).
2. `compute_scores(Q, K)`: Tính tích vô hướng $Q K^T$.
3. `scaled_dot_product_attention(Q, K, V, mask, temperature)`: Scaled dot-product attention hỗ trợ batch 3D, temperature và masking.

Chạy kiểm thử độc lập từ dòng lệnh:
```bash
python attention_numpy.py
```
**Kết quả kiểm tra:** Toàn bộ 7 unit tests (kiểm tra tính ổn định số học, Bài 1, 2, 3, 4, bài tập Scaling & Temperature, và kiểm tra batch shape 3D với tổng hàng bằng 1.0) đều đạt trạng thái `PASSED` 100%.

---

### 4.2. Mở và chạy Notebook `self_attention.ipynb`
Notebook được cấu trúc thành 13 phần khoa học, dễ tiếp cận cho sinh viên năm 2:
1. **Phần 1:** Import thư viện, kiểm tra môi trường và cấu hình Random Seed (`SEED=42`).
2. **Phần 2:** Nạp và khám phá tập dữ liệu `IMDB Dataset.csv` (loại bỏ 418 mẫu trùng lặp).
3. **Phần 3:** Tiền xử lý văn bản, tách từ, phân chia tập Train/Val/Test (stratified) và xây dựng `Vocabulary`.
4. **Phần 4:** Tạo biểu diễn ma trận đầu vào $X \in \mathbb{R}^{3 \times 2}$ cho ví dụ `["I", "like", "cats"]`.
5. **Phần 5:** Tính $Q, K, V$ qua các ma trận trọng số $W_Q, W_K, W_V$.
6. **Phần 6:** Tính ma trận điểm số Dot-product $S = Q K^T$.
7. **Phần 7:** Chuẩn hóa bằng Scaling $\frac{1}{\sqrt{d_k}}$ và hàm Softmax ổn định số học.
8. **Phần 8:** Tính ma trận đầu ra ngữ cảnh $O = A V$.
9. **Phần 9:** Kiểm tra tính hợp lệ kích thước tensor và kiểm định xác suất ($\sum \alpha = 1$).
10. **Phần 10:** Trực quan hóa ma trận Attention Matrix bằng Heatmap (Experiment 3: *"The cat that I adopted yesterday is sleeping."*).
11. **Phần 11:** Các thí nghiệm mở rộng: Thử nghiệm nhiệt độ Temperature $T \in \{0.5, 1.0, 2.0\}$ và Khảo sát chi phí bộ nhớ $O(T^2)$ theo Sequence Length ($T = 10 \to 10.000$).
12. **Phần 12:** Huấn luyện mô hình Self-Attention Classifier và so sánh định lượng với LSTM Baseline trên dữ liệu IMDB; vẽ Confusion Matrix và kiểm thử 3 câu ứng dụng tại Mục 26.
13. **Phần 13:** Phân tích lỗi (Error Analysis) và rút ra các quan sát mô hình.

Mọi cell trong notebook đã được chạy thực tế và nhúng sẵn kết quả, biểu đồ. Bạn có thể mở trực tiếp bằng VS Code hoặc Jupyter Notebook để kiểm tra.

---

## 5. Kết quả thực nghiệm chính

### 5.1. Bảng đối chiếu định lượng giữa LSTM và Self-Attention (`results.csv`)
Huấn luyện trên cùng một tập dữ liệu IMDB phân chia với `SEED=42`, trong 4 epochs trên CPU:

| Mô hình | Accuracy | Precision | Recall | F1-score | Thời gian huấn luyện (giây) | Số lượng tham số |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LSTM (Baseline LAB 05)** | 0.5480 | 0.5462 | 0.5680 | 0.5569 | 19.216 s | 997,505 |
| **Self-Attention Classifier** | **0.7240** | **0.7333** | **0.7040** | **0.7184** | **3.967 s** | 976,513 |

**Nhận xét khoa học:**
1. **Tốc độ huấn luyện:** Self-Attention hoàn tất huấn luyện trong **3.97 giây**, nhanh hơn gần **5 lần** so với LSTM (**19.22 giây**). Nguyên nhân là Self-Attention tận dụng tối đa phép nhân ma trận song song hóa trên toàn bộ sequence, loại bỏ vòng lặp tuần tự qua từng bước thời gian của mạng hồi quy.
2. **Hiệu quả phân loại:** Self-Attention đạt F1-score **71.84%** (so với LSTM chỉ đạt **55.69%** sau 4 epochs). Khả năng liên kết trực tiếp giữa các từ khóa cảm xúc ở các vị trí xa nhau trong câu giúp mô hình hội tụ nhanh hơn nhiều so với việc cố nén mọi thông tin qua chuỗi hidden state dài.

---

### 5.2. Khảo sát Temperature $T$ và Phân phối Attention
Thử nghiệm với score vector $s = [4.0, 2.0, 0.0]$:
- $T = 0.5$: Phân phối cực nhọn (Max weight: `0.9817`, Entropy: `0.0930`), tập trung gần như toàn bộ vào token có điểm số cao nhất.
- $T = 1.0$: Phân phối tiêu chuẩn (Max weight: `0.8668`, Entropy: `0.4411`).
- $T = 2.0$: Phân phối mượt và dàn đều (Max weight: `0.6652`, Entropy: `0.8324`), giảm độ phân hóa cực đoan.

---

### 5.3. Độ phức tạp tính toán và bộ nhớ theo Sequence Length $O(T^2)$
| Độ dài chuỗi ($T$) | Kích thước ma trận | Số lượng phần tử | Bộ nhớ lưu trữ (Float32) |
|:---:|:---:|---:|---:|
| **10** | $10 \times 10$ | 100 | $\approx 0.00$ MB |
| **100** | $100 \times 100$ | 10,000 | 0.04 MB |
| **500** | $500 \times 500$ | 250,000 | 0.95 MB |
| **1,000** | $1000 \times 1000$ | 1,000,000 | 3.81 MB |
| **10,000** | $10000 \times 1000$ | 100,000,000 | **381.47 MB** |

Khi độ dài chuỗi tăng 10 lần (từ 1,000 lên 10,000), số phần tử ma trận và bộ nhớ tăng gấp **100 lần** ($10^2$). Đây là hạn chế bản chất thúc đẩy các cải tiến như *FlashAttention*, *Sparse Attention* trong các mô hình ngôn ngữ lớn hiện đại.

---

## 6. Minh chứng hình ảnh và Phân tích lỗi
- **Heatmap Attention (`attention_heatmap.png`):** Trực quan hóa ma trận liên kết giữa các token trong câu cú pháp *"The cat that I adopted yesterday is sleeping."*, thể hiện rõ tương tác giữa Query và Key.
- **Confusion Matrix (`confusion_matrix.png`):** Ghi nhận phân phối dự đoán trên 1.000 mẫu kiểm thử (724 mẫu đúng, 276 mẫu sai).
- **Phân tích lỗi (`error_analysis.md`):** Đã phân tích chi tiết 3 trường hợp đúng điển hình và 3 trường hợp sai thực tế (như lỗi châm biếm, bẫy từ vựng cảm xúc, nhiễu nhãn dữ liệu IMDb và mệnh đề phủ định vế trước `"don't like... but interesting"`).

---
