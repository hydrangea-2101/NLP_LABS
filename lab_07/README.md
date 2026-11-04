# NLP LAB 07 — FROM ATTENTION TO TRANSFORMER
## 1. Tổng quan và Mục tiêu bài Lab
LAB 07 đánh dấu bước tiến từ cơ chế **Single-Head Self-Attention** (Lab 06) lên kiến trúc hoàn chỉnh của một khối **Transformer Encoder Block** (Vaswani et al., 2017 — *"Attention Is All You Need"*). Thay vì chỉ tính một ma trận Attention duy nhất và thiếu thông tin thứ tự từ, bài lab giải quyết trọn vẹn 4 mảnh ghép cốt lõi tạo nên sức mạnh của Transformer: **Multi-Head Self-Attention (MHA)**, **Sinusoidal Positional Encoding (PE)**, **Position-Wise Feed-Forward Network (FFN)**, cùng **Residual Connections & Layer Normalization**.

## 2. Cấu trúc thư mục Deliverables

Toàn bộ các tệp được tổ chức đầy đủ và chuẩn mực như sau:

```text
lab_07/
├── calculation_lab07.pdf                    # Báo cáo giải chi tiết 5 bài toán lý thuyết (Mục 6, 7, 10, 14, 15-16)
├── prediction.md                      # 6 câu hỏi dự đoán trước thực nghiệm (Mục 4) & đối chiếu kết quả (Mục 28)
├── positional_encoding.py             # Cài đặt Sinusoidal Positional Encoding & Permutation Experiment (Mục 9-12)
├── transformer_block.py               # Cài đặt từ gốc Multi-Head Attention, FFN & TransformerBlock (Mục 13-18)
├── transformer_classification.ipynb   # Notebook Jupyter 13 phần hoàn chỉnh, đã chạy và lưu toàn bộ kết quả
├── run_experiments.py                 # Script tự động chạy toàn bộ 6 thực nghiệm, xuất CSV, JSON và 4 biểu đồ
├── results.csv                        # Bảng số liệu đối chiếu định lượng đo đạc từ thực nghiệm
├── positional_encoding.png            # Biểu đồ ma trận Sinusoidal PE & sóng lượng giác đa tần số (Experiment 1)
├── attention_heads.png                # Biểu đồ Heatmap trực quan hóa 4 Attention Heads (Experiment 3)
├── learning_curves.png                # Đường cong huấn luyện Train/Val Loss & Accuracy (Mục 29)
├── confusion_matrix.png               # Ma trận nhầm lẫn của mô hình Transformer trên tập Test (1,000 mẫu)
├── error_analysis.md                  # Báo cáo phân tích sâu 3 ca đúng và 3 ca sai thực tế kèm bằng chứng Attention
├── reflection.md                      # Báo cáo phản tư học tập và tổng kết kiến trúc Transformer (Mục 27)
├── requirements.txt                   # Danh sách thư viện Python cần thiết
└── README.md                          # Hướng dẫn chi tiết toàn diện bài lab
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

### 4.1. Kiểm thử Sinusoidal Positional Encoding & Permutation Experiment (`positional_encoding.py`)
Tệp này cài đặt:
1. `compute_positional_encoding_numpy(max_len, d_model)`: Tính ma trận Sinusoidal PE bằng NumPy theo công thức $\sin\left(pos / 10000^{2i/d}\right)$ và $\cos\left(pos / 10000^{2i/d}\right)$.
2. `PositionalEncoding(d_model, max_len, dropout)`: Module PyTorch đăng ký `pe` làm `register_buffer` để cộng trực tiếp vào Token Embeddings.
3. Kiểm chứng bài tập tính tay **Mục 10** ($d_{\text{model}} = 4, pos = 0, 1, 2$), **Experiment 1 (Mục 11)** và **Permutation Experiment (Mục 12)** trên hai cặp câu đảo vị trí từ (`"I love NLP"` vs `"NLP love I"` và `"dog bites man"` vs `"man bites dog"`).

Chạy kiểm thử độc lập từ dòng lệnh:
```bash
python positional_encoding.py
```
**Kết quả kiểm tra:** Xác nhận khi không có PE, khoảng cách Euclid $L_2$ giữa hai câu đảo ngữ sau Mean Pooling bằng `0.00000000` tuyệt đối (bất biến hoán vị); khi bật Sinusoidal PE, khoảng cách $L_2$ tăng lên `0.395333` và `0.662406`, đồng thời tự động xuất biểu đồ `positional_encoding.png`.

---

### 4.2. Kiểm thử khối Transformer Encoder tự cài đặt (`transformer_block.py`)
Tệp này cài đặt hoàn toàn từ các lớp cơ bản (`nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `torch.matmul`) mà **không** sử dụng `nn.TransformerEncoderLayer`:
1. `MultiHeadSelfAttention(d_model, n_heads, dropout)`: Chiếu tuyến tính $Q, K, V$, tách thành $h$ đầu $(B, h, T, d_k)$, tính Scaled Dot-Product Attention có hỗ trợ Padding Mask, nối ghép $\text{Concat}(\text{head}_1, \dots, \text{head}_h)$ và nhân ma trận chiếu đầu ra $W^O$.
2. `PositionWiseFeedForward(d_model, d_ff, dropout)`: Cài đặt mạng phi tuyến 2 tầng $\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$.
3. `TransformerBlock(d_model, n_heads, d_ff, dropout)`: Kết hợp hoàn chỉnh MHA + Residual & LayerNorm 1 + FFN + Residual & LayerNorm 2.

Chạy kiểm thử độc lập từ dòng lệnh:
```bash
python transformer_block.py
```
**Kết quả kiểm tra:** Toàn bộ kiểm tra kích thước tensor tại **Mục 18** (`B = 8, T = 100, d_model = 128, n_heads = 4, d_ff = 256 => d_k = 32`), kiểm định tổng xác suất hàng Softmax bằng `1.0`, và kiểm thử các bài tập tính tay **Mục 6, 7, 14** đều đạt trạng thái `PASSED` 100%.

---

### 4.3. Mở và chạy Notebook `transformer_classification.ipynb`
Notebook được cấu trúc thành 13 phần khoa học, đồng nhất với phong cách trình bày của Lab 06:
1. **Phần 1:** Import thư viện, kiểm tra môi trường và cố định Random Seed (`SEED=42`).
2. **Phần 2:** Nạp và khám phá tập dữ liệu `IMDB Dataset.csv` (loại bỏ 418 mẫu trùng lặp, còn 49,582 mẫu).
3. **Phần 3:** Tiền xử lý văn bản, phân chia tập dữ liệu phân tầng (`5,000` Train / `1,000` Val / `1,000` Test) và xây dựng `Vocabulary` (`15,000` từ).
4. **Phần 4:** Kiểm chứng bằng code bài tập tính toán **Một Attention Head (Mục 6)** và **Multi-Head Concatenation (Mục 7)**.
5. **Phần 5:** Cài đặt và kiểm chứng bài tập tính toán **Sinusoidal Positional Encoding (Mục 9 & 10)** với $d_{\text{model}}=4, pos \in \{0, 1, 2\}$.
6. **Phần 6:** Thực nghiệm **Positional Encoding Visualization (Experiment 1 — Mục 11)** và **Permutation Experiment (Mục 12)** trên cặp câu `"I love NLP"` vs `"NLP love I"` và `"dog bites man"` vs `"man bites dog"`.
7. **Phần 7:** Kiểm chứng toán học **Position-Wise Feed-Forward Network (Mục 13 & 14)**, **Residual Connection và LayerNorm (Mục 15 & 16)**.
8. **Phần 8:** Cài đặt hoàn chỉnh **Transformer Encoder Block** và xác nhận bảng kiểm tra kích thước tensor ở mọi bước trung gian (**Mục 17 & 18**).
9. **Phần 9:** Thực hiện **Experiment 2 (Số lượng Attention Heads $h \in \{1, 2, 4, 8\}$)** và **Experiment 3 (Trực quan hóa độ đa dạng của 4 Attention Heads)** bằng Heatmap 4 ô.
10. **Phần 10:** Thực hiện **Experiment 4 (So sánh Transformer với LSTM Baseline & PyTorch API Reference)**, vẽ **Learning Curves (Mục 29)** và **Confusion Matrix**.
11. **Phần 11:** Thực hiện **Experiment 5 (Độ dài chuỗi $L \in \{32, 64, 128, 256\}$ & chi phí bậc hai $\mathcal{O}(T^2)$)** và **Experiment 6 (So sánh Mean Pooling vs `[CLS]`-style Pooling)**.
12. **Phần 12:** Ứng dụng mô hình phân loại cảm xúc trên 5 nhóm câu thực tế (**Mục 23**: câu ngắn, câu dài, phủ định, tương phản chuyển ý, cảm xúc pha trộn).
13. **Phần 13:** Phân tích lỗi (**Error Analysis — Mục 26**) trên 3 trường hợp dự đoán đúng và 3 trường hợp dự đoán sai kèm phân bổ trọng số Attention theo từng Head.

Mọi cell trong notebook đã được chạy thực tế và nhúng sẵn kết quả, bảng số liệu và biểu đồ trực quan.

---

## 5. Kết quả thực nghiệm chính

### 5.1. Experiment 2 — Ảnh hưởng của số lượng Attention Heads ($d_{\text{model}} = 128$ cố định)
Huấn luyện trên tập IMDb (`max_length = 128`, `4 epochs`, `batch_size = 64`):

| Số Heads ($h$) | Chiều mỗi Head ($d_k = d_{\text{model}}/h$) | Số tham số | Thời gian Train (giây) | Accuracy | Precision | Recall | F1-score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 Head** | 128 | 2,060,801 | **33.558 s** | 0.7900 | **0.8046** | 0.7660 | 0.7848 |
| **2 Heads** | 64 | 2,060,801 | 37.437 s | **0.7980** | 0.7854 | 0.8200 | 0.8023 |
| **4 Heads** | 32 | 2,060,801 | 46.895 s | 0.7920 | 0.7980 | 0.7820 | 0.7899 |
| **8 Heads** | 16 | 2,060,801 | 65.255 s | 0.7830 | 0.7331 | **0.8900** | **0.8040** |

**Nhận xét khoa học:**
1. **Bất biến số lượng tham số:** Khi cố định $d_{\text{model}} = 128$, việc chia thành $h \in \{1, 2, 4, 8\}$ heads với $d_k = 128/h$ giữ nguyên tổng số tham số của toàn mạng ở mức **2,060,801 tham số** vì $h \times (d_{\text{model}} \times d_k) = d_{\text{model}}^2$.
2. **Đánh đổi giữa số lượng không gian con và chiều biểu diễn mỗi Head:** Chuyển từ `1 Head` sang `2 Heads` và `4 Heads` giúp tăng F1-score từ `0.7848` lên `0.8023` nhờ khả năng theo dõi đồng thời nhiều quan hệ ngữ nghĩa. Tuy nhiên, khi tăng lên `8 Heads` ($d_k = 16$), không gian con mỗi Head trở nên quá hẹp khiến Accuracy bắt đầu giảm nhẹ xuống `0.7830` và thời gian tính toán tăng lên `65.26s` do chi phí quản lý $B \times h$ ma trận $T \times T$.

---

### 5.2. Experiment 4 — Đối chiếu định lượng giữa LSTM và Transformer (`results.csv`)
Huấn luyện công bằng với cùng `vocab_size = 15,000`, `d_model = 128`, `max_length = 128`, `4 epochs` trên CPU:

| Mô hình | Accuracy | Precision | Recall | F1-score | Thời gian huấn luyện (giây) | Số lượng tham số |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LSTM Baseline (Model A)** | 0.6880 | 0.6992 | 0.6600 | 0.6790 | 91.774 s | 2,060,417 |
| **Transformer Custom Block (Model B)** | **0.7920** | **0.7980** | **0.7820** | **0.7899** | **46.895 s** | 2,060,801 |
| **Transformer PyTorch API Reference** | **0.8020** | **0.8133** | **0.7840** | **0.7984** | 54.483 s | 2,060,801 |

**Nhận xét khoa học:**
1. **Tốc độ và khả năng song song hóa:** Dù có số lượng tham số tương đương nhau ($\approx 2.06$ triệu tham số), khối `TransformerBlock` tự cài đặt hoàn tất 4 epochs trong **46.90 giây**, nhanh gấp **1.96 lần** so với LSTM (**91.77 giây**) nhờ loại bỏ hoàn toàn vòng lặp tuần tự theo bước thời gian $t = 1 \dots T$.
2. **Chất lượng biểu diễn:** Transformer vượt trội hơn LSTM **+10.40% Accuracy** (`79.20%` so với `68.80%`) và **+11.09% F1-score** (`0.7899` so với `0.6790`). Đồng thời, module `TransformerBlock` tự xây dựng từ gốc đạt hiệu năng tiệm cận sát sao với `nn.TransformerEncoderLayer` chuẩn của PyTorch (`0.7899` vs `0.7984`).

---

### 5.3. Experiment 5 & 6 — Độ phức tạp bậc hai $\mathcal{O}(T^2)$ theo Sequence Length và Chiến lược Pooling

**Bảng khảo sát Sequence Length $L \in \{32, 64, 128, 256\}$ (`n_heads = 4`):**

| Độ dài tối đa ($L$) | Tỷ lệ Token giữ lại | Số phần tử $T^2$ / Head | Bộ nhớ Attention / Batch (Float32) | Thời gian Train (giây) | Accuracy | F1-score |
|:---:|:---:|---:|---:|---:|:---:|:---:|
| **32** | 21.5% (32.0 từ) | 1,024 | 1.00 MB | **10.900 s** | 0.6410 | 0.6679 |
| **64** | 41.3% (63.2 từ) | 4,096 | 4.00 MB | 21.510 s | 0.7300 | 0.7554 |
| **128** | 70.1% (117.7 từ) | 16,384 | 16.00 MB | 46.895 s | 0.7920 | 0.7899 |
| **256** | **90.5%** (175.6 từ) | **65,536** | **64.00 MB** | 166.221 s | **0.8300** | **0.8346** |

**Bảng đối chiếu Chiến lược Pooling (`max_length = 128`, `4 epochs`):**

| Chiến lược Pooling | Accuracy | Precision | Recall | F1-score | Thời gian Train (giây) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Masked Mean Pooling** | **0.7920** | **0.7980** | 0.7820 | **0.7899** | **46.895 s** |
| **First-token (`[CLS]`-style) Pooling** | 0.7840 | 0.7773 | **0.7960** | 0.7866 | 47.079 s |

**Nhận xét khoa học:**
- Khi tăng $L$ gấp đôi từ `128` lên `256`, số phần tử trong ma trận Attention $T^2$ và dung lượng bộ nhớ tăng gấp **4 lần** ($16,384 \to 65,536$ phần tử; $16\text{ MB} \to 64\text{ MB}$), kéo thời gian huấn luyện tăng gấp **3.54 lần** (`46.90s` $\to$ `166.22s`), nhưng đổi lại giữ được **90.5%** nội dung bài đánh giá và đẩy F1-score lên mức cao nhất **83.46%**.
- Khi huấn luyện từ đầu (from scratch) chỉ với 1 tầng Encoder, **Masked Mean Pooling** đạt độ ổn định cao hơn nhẹ so với **`[CLS]`-style Pooling** (`79.20%` vs `78.40%` Accuracy) vì gradient từ hàm mất mát được phân bổ trực tiếp về tất cả các vị trí token hợp lệ.

---

## 6. Minh chứng hình ảnh và Phân tích lỗi
- **Biểu đồ Positional Encoding (`positional_encoding.png`):** Minh họa trực quan bản đồ nhiệt Sinusoidal PE ($100 \times 128$) và các sóng hình sin/cosin giảm dần tần số theo cấp số nhân qua các chiều `dim 0, 1, 16, 32, 64`.
- **Heatmap 4 Attention Heads (`attention_heads.png`):** Minh chứng sự phân hóa vai trò rõ rệt giữa 4 Heads trên câu *"the movie was not good at first but the ending was truly great"* — từ Head cục bộ sắc nhọn (`Entropy = 1.33`, liên kết `'great' -> 'great'`) đến các Head ngữ cảnh kết nối phủ định và tương phản xa (`'not' -> 'ending'`, `'movie' -> 'good'`, `'the' -> 'truly'`).
- **Đường cong học tập (`learning_curves.png`) & Confusion Matrix (`confusion_matrix.png`):** Minh họa tốc độ hội tụ nhanh của Transformer so với LSTM và phân phối dự đoán cân bằng trên 1,000 mẫu Test (`TN = 401, FP = 99, FN = 109, TP = 391`).
- **Phân tích lỗi (`error_analysis.md`):** Phân tích chi tiết 3 ca đúng và 3 ca sai thực tế trên tập Test kèm trọng số Attention của từng Head, làm rõ các nguyên nhân cốt lõi: hiện tượng cắt cụt đoạn kết tại `max_length = 128`, nhiễu từ vựng cảm xúc khi so sánh đối chiếu bộ phim khác, giọng văn châm biếm/ẩn dụ (*"clay pigeon material"*) và thảo luận nguyên lý **"Attention $\neq$ Explanation"** (Jain & Wallace, 2019).

---
